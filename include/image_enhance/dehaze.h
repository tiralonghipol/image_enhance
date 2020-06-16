#ifndef HAZE_REMOVAL_H
#define HAZE_REMOVAL_H

#include "guided_filter.h"
#include "opencv2/opencv.hpp"

#include <vector>


namespace dehaze
{
	typedef struct _pixel
	{
		int i;
		int j;
		uchar val;
		_pixel(int _i, int _j, uchar _val) : i(_i), j(_j), val((uchar)_val) {}
		bool operator<(const _pixel a) const
		{
			return a.val < val;
		}
	} Pixel;

	class CHazeRemoval
	{
	public:
		CHazeRemoval(const cv::Mat *in_frame,
						double omega, double t0, int radius, int r, double eps);
		~CHazeRemoval();
		void Process(cv::Mat &out_frame, bool use_old_alg=false);
	private:
		void get_dark_channel();
		void get_dark_channel_old();
		void get_air_light();
		void get_transmission();
		void get_transmission_old();
		void guided_filter();
		void recover();

		int m_rows;
		int m_cols;
		int m_channels;

		int m_radius;
		int m_r;

		double m_omega;
		double m_t0;
		double m_eps;

		cv::Mat m_in_frame;

		cv::Vec3d *m_vec3d_avg_light_ptr = NULL;
		std::vector<Pixel> m_light_pixel_vect;

		cv::Mat *m_cv_mat_transmission_ptr = NULL;
		cv::Mat *m_cv_mat_guided_filter_output_ptr = NULL;
		cv::Mat *m_cv_mat_recover_result_ptr = NULL;
	};
} // namespace dehaze
#endif // !HAZE_REMOVAL_H