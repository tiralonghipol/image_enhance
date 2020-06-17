#include "ros/ros.h"
#include "image_enhance/dehaze.h"
#include <iostream>

#define GET_INDEX(x, y, cols)  (x * cols) + y
#define GET_MIN_VAL(r, g, b)   b > g ? ((g > r) ? r : g) : ((b > r) ? r : b)
#define GET_MIN(a,b)   a > b ? b : a
#define GET_MAX(a,b)   a < b ? b : a

#define SHOW_IMAGE_STEPS false

namespace dehaze
{
	
	static cv::Mat boxfilter(const cv::Mat &I, int r)
	{
		cv::Mat result;
		cv::blur(I, result, cv::Size(r, r));
		return result;
	}
	CHazeRemoval::CHazeRemoval(const cv::Mat *in_frame,
						double omega, double t0, int radius, int r, double eps, int filter_resize_factor)
	{
		m_rows = in_frame->rows;
		m_cols = in_frame->cols;
		m_channels = in_frame->channels();
		m_omega = omega;
		m_t0 = t0;
		m_radius = radius;
		m_r = r;
		m_eps = eps;
		m_filter_resize_factor = filter_resize_factor;
		m_light_pixel_vect.reserve(m_rows*m_cols);

		m_in_frame = in_frame->clone();

		m_vec3d_avg_light_ptr = new cv::Vec3d();
		m_cv_mat_transmission_ptr = new cv::Mat(m_rows, m_cols, CV_64FC1);
		m_cv_mat_guided_filter_output_ptr = new cv::Mat(m_rows, m_cols, CV_64FC1);
		m_cv_mat_recover_result_ptr = new cv::Mat(m_rows, m_cols, CV_64FC3);
	}
	CHazeRemoval::~CHazeRemoval()
	{
		if(m_cv_mat_guided_filter_output_ptr != NULL)
		{
			delete m_cv_mat_guided_filter_output_ptr;
		}
		if(m_cv_mat_recover_result_ptr != NULL)
		{
			delete m_cv_mat_recover_result_ptr;
		}
		if(m_cv_mat_transmission_ptr != NULL)
		{
			delete m_cv_mat_transmission_ptr;
		}
		if(m_vec3d_avg_light_ptr != NULL)
		{
			delete m_vec3d_avg_light_ptr;
		}
	}
	void CHazeRemoval::Process(cv::Mat &out_frame, bool use_old_alg)
	{
		#if SHOW_IMAGE_STEPS
		imshow("Original Image", m_in_frame);
		cv::waitKey(1);
		#endif

		clock_t start = clock();
		clock_t proc_start = start;

		if(use_old_alg) get_dark_channel_old();
		else get_dark_channel();

		get_air_light();

		if(use_old_alg) get_transmission_old();
		else get_transmission();
		
		#if SHOW_IMAGE_STEPS
		imshow("Transmission", *m_cv_mat_transmission_ptr);
		cv::waitKey(1);
		#endif

		guided_filter();

		#if SHOW_IMAGE_STEPS
		imshow("Guided Filter", *m_cv_mat_guided_filter_output_ptr);
		cv::waitKey(1);
		#endif

		recover();

		m_cv_mat_recover_result_ptr->convertTo(out_frame, CV_8UC3);
	}
	void CHazeRemoval::get_dark_channel_old()
	{
		// old algorithm (50mSec per frame)
		for (int i = 0; i < m_rows; i++)
		{
			for (int j = 0; j < m_cols; j++)
			{
				int rmin = cv::max(0, i - m_radius);
				int rmax = cv::min(i + m_radius, m_rows - 1);
				int cmin = cv::max(0, j - m_radius);
				int cmax = cv::min(j + m_radius, m_cols - 1);
				double min_val = 255.0;
				for (int x = rmin; x <= rmax; x++)
				{
					for (int y = cmin; y <= cmax; y++)
					{
						cv::Vec3b tmp = m_in_frame.ptr<cv::Vec3b>(x)[y];
						uchar b = tmp[0];
						uchar g = tmp[1];
						uchar r = tmp[2];
						uchar minpixel = GET_MIN_VAL(r,g,b);
						min_val = cv::min((double)minpixel, min_val);
					}
				}
				m_light_pixel_vect.push_back(Pixel(i, j, uchar(min_val)));
			}
		}
	}
	void CHazeRemoval::get_dark_channel()
	{
		/* 
		old algorithm (~ 50ms per frame)
			1. Iterate over each pixel, iterating over their neighborhood, find smallest value and save it
			2. Sort the resulting smallest pixel value vector
		new algorithm (~ 10ms per frame)
			1. Iterate over each pixel and save the smallest rgb vals for each pixel
			2. Iterate over each pixel, Iterate over that pixel's row neighbors, save the smallest value
			3. Iterate over each pixel, Iterate over that pixel's column neighbors, save the smallest value
			4. Sort the resulting smallest pixel value vector
			... but the results aren't the same...
		*/
		// create an array to store the minimum pixel values (min of each rgb)
		double *pix_mins = new double[m_rows*m_cols];
		unsigned long px_idx = 0;
		// for each pixel, calculate the min rgb
		for (int i = 0; i < m_rows; ++i)
		{
			for (int j = 0; j < m_cols; ++j)
			{
				// get the vector of pixel values
				cv::Vec3b tmp = m_in_frame.ptr<cv::Vec3b>(i)[j];
				// save the smallest value at the corresponding array location
				pix_mins[px_idx] = GET_MIN_VAL(double(tmp[0]), double(tmp[1]), double(tmp[2]));
				px_idx++;
			}
		}
		// create an array to store the min row values (min of each row of neighbors)
		double *row_mins = new double[m_rows*m_cols];
		px_idx = 0;
		// for each pixel, calculate the min of the row neighbors
		for (int i = 0; i < m_rows; ++i)
		{
			// calculate the neighbor bounds on the row neighbors
				int rmin = cv::max(0, i - m_radius);
				int rmax = cv::min(i + m_radius, m_rows - 1);
			// loop through the columns
			for (int j = 0; j < m_cols; ++j)
			{
				double min_val = 255.0;
				// loop over the bytes in the row
				for(int k = rmin; k <= rmax; ++k)
				{
					min_val = cv::min((double)pix_mins[GET_INDEX(k, j, m_cols)], min_val);
				}
				row_mins[px_idx] = min_val;
				px_idx++;
			}
		}
		m_light_pixel_vect.reserve(m_rows * m_cols);
		// for each pixel, calculate the min of the column neighbors
		for (int i = 0; i < m_rows; ++i)
		{
			for (int j = 0; j < m_cols; ++j)
			{ 
				int cmin = cv::max(0, j - m_radius);
				int cmax = cv::min(j + m_radius, m_cols - 1);
				double min_val = 255.0;
				// iterate over the column
				for(int k = cmin; k <= cmax; ++k)
				{
					min_val = cv::min((double)pix_mins[GET_INDEX(i, k, m_cols)], min_val);
				}
				m_light_pixel_vect.push_back(Pixel(i, j, min_val));
			}
		}
		delete row_mins;
		delete pix_mins;
	}

	// This function is pretty fast... gets the average of the top few pixels (0.1%) of m_light_pixel_vect
	void CHazeRemoval::get_air_light()
	{
		std::sort(m_light_pixel_vect.begin(), m_light_pixel_vect.end());
		int num = int(m_rows * m_cols * 0.001);
		double A_sum[3] = {0.0,0.0,0.0};
		std::vector<Pixel>::iterator it = m_light_pixel_vect.begin();
		for (int cnt = 0; cnt < num; cnt++)
		{
			cv::Vec3b tmp = m_in_frame.ptr<cv::Vec3b>(it->i)[it->j];
			A_sum[0] += tmp[0];
			A_sum[1] += tmp[1];
			A_sum[2] += tmp[2];
			it++;
		}
		for (int i = 0; i < 3; i++)
		{
			(*m_vec3d_avg_light_ptr)[i] = A_sum[i] / num;
		}
	}
	// this function is slow... (200ms?!?!?)
	void CHazeRemoval::get_transmission_old()
	{
		for (int i = 0; i < m_rows; i++)
		{
			for (int j = 0; j < m_cols; j++)
			{
				int rmin = cv::max(0, i - m_radius);
				int rmax = cv::min(i + m_radius, m_rows - 1);
				int cmin = cv::max(0, j - m_radius);
				int cmax = cv::min(j + m_radius, m_cols - 1);
				double min_val = 255.0;
				for (int x = rmin; x <= rmax; x++)
				{
					for (int y = cmin; y <= cmax; y++)
					{
						cv::Vec3b tmp = m_in_frame.ptr<cv::Vec3b>(x)[y];
						double b = (double)tmp[0] / (*m_vec3d_avg_light_ptr)[0];
						double g = (double)tmp[1] / (*m_vec3d_avg_light_ptr)[1];
						double r = (double)tmp[2] / (*m_vec3d_avg_light_ptr)[2];
						double minpixel = b > g ? ((g > r) ? r : g) : ((b > r) ? r : b);
						min_val = cv::min(minpixel, min_val);
					}
				}
				m_cv_mat_transmission_ptr->ptr<double>(i)[j] = 1 - m_omega * min_val;
			}
		}
	}
	void CHazeRemoval::get_transmission()
	{
		// pre calculate the minimum pixel values for each pixel, so they're calculated n times, not n*m times
		std::vector<double> pix_mins;
		pix_mins.reserve(m_rows*m_cols);
		for (int i = 0; i < m_rows; ++i)
		{
			for (int j = 0; j < m_cols; ++j)
			{
				cv::Vec3b tmp = m_in_frame.ptr<cv::Vec3b>(i)[j];
				double b = (double)tmp[0] / (*m_vec3d_avg_light_ptr)[0];
				double g = (double)tmp[1] / (*m_vec3d_avg_light_ptr)[1];
				double r = (double)tmp[2] / (*m_vec3d_avg_light_ptr)[2];
				pix_mins.push_back(GET_MIN_VAL(b,g,r));
			}
		}
		// create an array to store the min row values (min of each row of neighbors)
		std::vector<double> row_mins;
		row_mins.reserve(m_rows*m_cols);
		// for each pixel, calculate the min of the row neighbors
		for (int i = 0; i < m_rows; ++i)
		{
			for (int j = 0; j < m_cols; ++j)
			{
				int min_i = std::max(0, i-m_radius);
				int max_i = std::min(m_rows, i+m_radius);
				int px_idx = GET_INDEX(i, j, m_cols);
				double row_min = 255.0;
				// loop over the bytes in the row
				for(int k = min_i; k < max_i; ++k)
				{
					row_min = GET_MIN(pix_mins[GET_INDEX(k, j, m_cols)], row_min);
				}
				row_mins.push_back(row_min);
			}
		}
		// for each pixel, calculate the min of the column neighbors
		for (int i = 0; i < m_rows; ++i)
		{
			for (int j = 0; j < m_cols; ++j)
			{
				int min_j = std::max(0, j-m_radius);
				int max_j = std::min(m_cols, j+m_radius);
				double min_val = 255.0;
				// iterate over the column
				for(int k = min_j; k < max_j; ++k)
				{
					int idx = GET_INDEX(i, k, m_cols);
					min_val = GET_MIN(row_mins[idx], min_val);
				}
				m_cv_mat_transmission_ptr->ptr<double>(i)[j] = 1 - m_omega * min_val;
			}
		}	
	}
	void CHazeRemoval::guided_filter()
	{
		*m_cv_mat_guided_filter_output_ptr = guidedFilter(m_in_frame, *m_cv_mat_transmission_ptr, m_r, m_eps, m_filter_resize_factor);
	}
	void CHazeRemoval::recover()
	{
		for (int i = 0; i < m_rows; i++)
		{
			for (int j = 0; j < m_cols; j++)
			{
				for (int c = 0; c < m_channels; c++)
				{
					// ((rgb - average light val) / guidedfilter) + average light val
					double val = (double(m_in_frame.ptr<cv::Vec3b>(i)[j][c]) - (*m_vec3d_avg_light_ptr)[c]) / 
								cv::max(m_t0, m_cv_mat_guided_filter_output_ptr->ptr<double>(i)[j]) + (*m_vec3d_avg_light_ptr)[c];
					m_cv_mat_recover_result_ptr->ptr<cv::Vec3d>(i)[j][c] = cv::max(0.0, cv::min(255.0, val));
				}
			}
		}
	}
} // namespace dehaze