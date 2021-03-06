#include "image_enhance/clahe.h"

namespace clahe
{

	ClaheRos::ClaheRos()
	{
	}

	// void ClaheRos::imageCb(const sensor_msgs::ImageConstPtr &msg)
	cv::Mat ClaheRos::Process(const cv::Mat image_input)
	{
		double clahe_clip_limit = _clahe_clip_limit;
		double clahe_grid_size = _clahe_grid_size;
		// std_msgs::Header header = msg->header;
		// cv::Mat bgr_image = cv_bridge::toCvShare(msg, "bgr8")->image;
		cv::Mat bgr_image = image_input;
		cv::Mat lab_image, dst;
		cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

		// Extract the L channel
		std::vector<cv::Mat> lab_planes(3);
		cv::split(lab_image, lab_planes); // now we have the L image in lab_planes[0]

		// apply the CLAHE algorithm to the L channel
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(clahe_clip_limit);
		clahe->setTilesGridSize(cv::Size(clahe_grid_size, clahe_grid_size));
		clahe->apply(lab_planes[0], dst);

		// Merge the the color planes back into an Lab image
		dst.copyTo(lab_planes[0]);
		cv::merge(lab_planes, lab_image);

		// convert back to RGB
		cv::Mat image_clahe;
		cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
		// image_pub_.publish(cv_bridge::CvImage(header, "bgr8", image_clahe).toImageMsg());
		return image_clahe;
	}
} // namespace clahe