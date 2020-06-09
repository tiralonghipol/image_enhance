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
	} Pixel;

	int val = 500;

	class CHazeRemoval
	{
	public:
		CHazeRemoval();
		~CHazeRemoval();

	public:
		bool InitProc(int width, int height, int nChannels);
		bool Process(const unsigned char *indata, unsigned char *outdata, int width, int height, int nChannels);

	private:
		int rows;
		int cols;
		int channels;
	};

	int _radius;
	double _omega;
	double _t0;
	int _r;
	double _eps;

	void get_dark_channel(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, int rows, int cols, int channels, int radius);
	void get_air_light(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, cv::Vec3d *p_Alight, int rows, int cols, int channels);
	void get_transmission(const cv::Mat *p_src, cv::Mat *p_tran, cv::Vec3d *p_Alight, int rows, int cols, int channels, int radius, double omega);
	void guided_filter(const cv::Mat *p_src, const cv::Mat *p_tran, cv::Mat *p_gtran, int r, double eps);
	void recover(const cv::Mat *p_src, const cv::Mat *p_gtran, cv::Mat *p_dst, cv::Vec3d *p_Alight, int rows, int cols, int channels, double t0);
	void assign_data(unsigned char *outdata, const cv::Mat *p_dst, int rows, int cols, int channels);
	void get_min_vect(const uint8_t *pix_mins, std::vector<Pixel> &tmp_vec, uint16_t rows, uint16_t cols, uint16_t radius);
} // namespace dehaze
#endif // !HAZE_REMOVAL_H