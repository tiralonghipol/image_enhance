#include "image_enhance/dehaze.h"
#include <iostream>

#define OLD_DARK_ALG true
#define OLD_TRANS_ALG true

using namespace cv;
using namespace std;
namespace dehaze
{
	CHazeRemoval::CHazeRemoval()
	{
		rows = 0;
		cols = 0;
		channels = 0;
	}

	CHazeRemoval::~CHazeRemoval()
	{
	}

	bool CHazeRemoval::InitProc(int width, int height, int nChannels)
	{
		bool ret = false;
		rows = height;
		cols = width;
		channels = nChannels;

		if (width > 0 && height > 0 && nChannels == 3)
			ret = true;
		return ret;
	}

	bool CHazeRemoval::Process(const unsigned char *indata, unsigned char *outdata, int width, int height, int nChannels)
	{
		bool ret = true;
		if (!indata || !outdata)
		{
			ret = false;
		}
		rows = height;
		cols = width;
		channels = nChannels;

		int radius = _radius;
		double omega = _omega;
		double t0 = _t0;
		int r = _r;
		double eps = _eps;

		vector<Pixel> tmp_vec;
		Mat *p_src = new Mat(rows, cols, CV_8UC3, (void *)indata);
		Mat *p_dst = new Mat(rows, cols, CV_64FC3);
		//Mat * p_dark = new Mat(rows, cols, CV_64FC1);
		Mat *p_tran = new Mat(rows, cols, CV_64FC1);
		Mat *p_gtran = new Mat(rows, cols, CV_64FC1);
		Vec3d *p_Alight = new Vec3d();

		clock_t start = clock();
		clock_t proc_start = start;
		get_dark_channel(p_src, tmp_vec, rows, cols, channels, radius);
		std::cout << "Dark Channel proc time: " << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
		start = clock();
		get_air_light(p_src, tmp_vec, p_Alight, rows, cols, channels);
		std::cout << "Get Air Light proc time: " << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
		start = clock();
		get_transmission(p_src, p_tran, p_Alight, rows, cols, channels, radius = 7, omega);
		std::cout << "Get Transmission proc time: " << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
		start = clock();
		guided_filter(p_src, p_tran, p_gtran, r, eps);
		std::cout << "Guided Filter proc time: " << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
		start = clock();
		recover(p_src, p_gtran, p_dst, p_Alight, rows, cols, channels, t0);
		std::cout << "Recover proc time: " << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
		start = clock();
		assign_data(outdata, p_dst, rows, cols, channels);
		std::cout << "Assign Data proc time: " << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
		std::cout << "Total proc time: " << (float)(clock() - proc_start) / CLOCKS_PER_SEC << std::endl;

		return ret;
	}

	bool sort_fun(const Pixel &a, const Pixel &b)
	{
		return a.val > b.val;
	}
	void get_dark_channel(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, int rows, int cols, int channels, int radius)
	{
		#if OLD_DARK_ALG
		// old algorithm (50mSec per frame)
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				int rmin = cv::max(0, i - radius);
				int rmax = cv::min(i + radius, rows - 1);
				int cmin = cv::max(0, j - radius);
				int cmax = cv::min(j + radius, cols - 1);
				double min_val = 255;
				for (int x = rmin; x <= rmax; x++)
				{
					for (int y = cmin; y <= cmax; y++)
					{
						cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(x)[y];
						uchar b = tmp[0];
						uchar g = tmp[1];
						uchar r = tmp[2];
						uchar minpixel = b > g ? ((g > r) ? r : g) : ((b > r) ? r : b);
						min_val = cv::min((double)minpixel, min_val);
					}
				}
				//p_dark->ptr<double>(i)[j] = min_val;
				tmp_vec.push_back(Pixel(i, j, uchar(min_val)));
			}
		}
		#else
		// new algorithm (8ms per frame)
		// create an array to store the minimum pixel values (min of each rgb)
		double pix_mins[rows*cols];
		// for each pixel, calculate the min rgb
		std::cout <<"calc pix mins" << std::endl;
		for (uint16_t i = 0; i < rows; ++i)
		{
			for (uint16_t j = 0; j < cols; ++j)
			{
				uint16_t px_idx = (j*cols) + i;
				cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(i)[j];
				pix_mins[px_idx] = std::min({(double)tmp[0], (double)tmp[1], (double)tmp[2], (double)255});
			}
		}
		// create an array to store the min row values (min of each row of neighbors)
		double row_mins[rows*cols];
		std::cout <<"calc row mins" << std::endl;
		// for each pixel, calculate the min of the row neighbors
		for (int i = 0; i < rows; ++i)
		{
			uint16_t min_i = std::max(0,    i-radius);
			uint16_t max_i = std::min(cols, i+radius);
			for (int j = 0; j < cols; ++j)
			{
				uint16_t px_idx = (j*cols) + i;
				row_mins[px_idx] = 255.0;
				// loop over the bytes in the row
				for(uint16_t k = min_i; k < max_i; ++k)
				{
					row_mins[px_idx] = std::min(pix_mins[(j*cols) + k], row_mins[px_idx]);
				}
			}
		}
		// for each pixel, calculate the min of the column neighbors
		std::cout <<"calc col mins" << std::endl;
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				uint16_t min_j = std::max(0,    j - radius);
				uint16_t max_j = std::min(cols, j + radius);
				double min_val = 255.0;
				// iterate over the column
				for(uint16_t k = min_j; k < max_j; ++k)
				{
					uint16_t idx = k*cols + i;
					min_val = std::min(row_mins[idx], min_val);
				}
				tmp_vec.push_back(Pixel(i, j, uchar(min_val)));
			}
		}	
		#endif
		std::sort(tmp_vec.begin(), tmp_vec.end(), sort_fun);
	}

	// This function is pretty fast... gets the average of the top few pixels (0.1%) of tmp_vec
	void get_air_light(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, cv::Vec3d *p_Alight, int rows, int cols, int channels)
	{
		int num = int(rows * cols * 0.001);
		double A_sum[3] = {
			0,
		};
		std::vector<Pixel>::iterator it = tmp_vec.begin();
		for (int cnt = 0; cnt < num; cnt++)
		{
			cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(it->i)[it->j];
			A_sum[0] += tmp[0];
			A_sum[1] += tmp[1];
			A_sum[2] += tmp[2];
			it++;
		}
		for (int i = 0; i < 3; i++)
		{
			(*p_Alight)[i] = A_sum[i] / num;
		}
	}
	// this function is slow... (200ms?!?!?)
	void get_transmission(const cv::Mat *p_src, cv::Mat *p_tran, cv::Vec3d *p_Alight, int rows, int cols, int channels, int radius, double omega)
	{
		#if OLD_TRANS_ALG
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					int rmin = cv::max(0, i - radius);
					int rmax = cv::min(i + radius, rows - 1);
					int cmin = cv::max(0, j - radius);
					int cmax = cv::min(j + radius, cols - 1);
					double min_val = 255.0;
					for (int x = rmin; x <= rmax; x++)
					{
						for (int y = cmin; y <= cmax; y++)
						{
							cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(x)[y];
							double b = (double)tmp[0] / (*p_Alight)[0];
							double g = (double)tmp[1] / (*p_Alight)[1];
							double r = (double)tmp[2] / (*p_Alight)[2];
							double minpixel = b > g ? ((g > r) ? r : g) : ((b > r) ? r : b);
							min_val = cv::min(minpixel, min_val);
						}
					}
					p_tran->ptr<double>(i)[j] = 1 - omega * min_val;
				}
			}
		#else
			// pre calculate the minimum pixel values for each pixel, so they're not calculated num_neighbor^2 times
			double pix_mins[rows*cols];
			for (uint16_t i = 0; i < rows; i++)
			{
				for (uint16_t j = 0; j < cols; j++)
				{
					// calculate the linear index of the pixel
					uint16_t px_idx = ((i*cols) + j);
					// get a pointer to the pixel
					//uint8_t* ptr = p_src->data + (px_idx * p_src->channels());
					cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(i)[j];
					pix_mins[px_idx] = (double)tmp[0] / (*p_Alight)[0]; 
					pix_mins[px_idx] = std::min((double)tmp[1] / (*p_Alight)[1], (double)pix_mins[px_idx]);
					pix_mins[px_idx] = std::min((double)tmp[2] / (*p_Alight)[2], (double)pix_mins[px_idx]);
				}
			}
			// create an array to store the min row values (min of each row of neighbors)
			double row_mins[rows*cols];
			// for each pixel, calculate the min of the row neighbors
			for (int row = 0; row < rows; ++row)
			{
				for (int col = 0; col < cols; ++col)
				{
					uint16_t min_row = std::max(0, row-radius);
					uint16_t max_row = std::min(rows, row+radius);
					uint16_t px_idx = (row*cols) + col;
					row_mins[px_idx] = 255.0;
					// loop over the bytes in the row
					for(uint16_t k = min_row; k < max_row; ++k)
					{
						uint16_t neighbor_idx = (k*cols) + col;
						row_mins[px_idx] = std::min(pix_mins[neighbor_idx], row_mins[px_idx]);
					}
				}
			}
			// for each pixel, calculate the min of the column neighbors
			for (uint16_t row = 0; row < rows; ++row)
			{
				for (uint16_t col = 0; col < cols; ++col)
				{
					uint16_t min_col = std::max(0, col-radius);
					uint16_t max_col = std::min(cols, col+radius);
					double min_val = 255.0;
					// iterate over the column
					for(uint16_t k = min_col; k < max_col; ++k)
					{
						uint16_t idx = k*cols + col;
						min_val = std::min(row_mins[idx], min_val);
					}
					p_tran->ptr<double>(row)[col] = 1 - omega * min_val;
				}
			}	
		#endif

	}
	void guided_filter(const cv::Mat *p_src, const cv::Mat *p_tran, cv::Mat *p_gtran, int r, double eps)
	{
		*p_gtran = guidedFilter(*p_src, *p_tran, r, eps);
	}

	void recover(const cv::Mat *p_src, const cv::Mat *p_gtran, cv::Mat *p_dst, cv::Vec3d *p_Alight, int rows, int cols, int channels, double t0)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				for (int c = 0; c < channels; c++)
				{
					double val = (double(p_src->ptr<cv::Vec3b>(i)[j][c]) - (*p_Alight)[c]) / cv::max(t0, p_gtran->ptr<double>(i)[j]) + (*p_Alight)[c];
					p_dst->ptr<cv::Vec3d>(i)[j][c] = cv::max(0.0, cv::min(255.0, val));
				}
			}
		}
	}

	void assign_data(unsigned char *outdata, const cv::Mat *p_dst, int rows, int cols, int channels)
	{
		for (int i = 0; i < rows * cols * channels; i++)
		{
			*(outdata + i) = (unsigned char)(*((double *)(p_dst->data) + i));
		}
	}
} // namespace dehaze