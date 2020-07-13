#pragma once

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <vector>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <std_msgs/Header.h>

namespace clahe
{
	class ClaheRos
	{
	public:
		ClaheRos();
		cv::Mat Process(const cv::Mat image_input);

	};
	double _clahe_clip_limit;
	double _clahe_grid_size;

} // namespace clahe
