#include <cstdio>
#include <cstdlib>
#include "image_enhance/image_enhance.h"
#include "image_enhance/dehaze.h"
#include "image_enhance/guided_filter.h"

imageEnhance::imageEnhance(ros::NodeHandle &n, const std::string &s, int bufSize)
{

	// parameters
	n.getParam("enable_dyn_reconf", _enable_dyn_reconf);
	n.getParam("topic_image_input", _topic_image_input);
	n.getParam("topic_image_output", _topic_image_output);
	n.getParam("use_dehaze", _use_dehaze);
	n.getParam("scale_factor", _scale_factor);

	image_transport::ImageTransport it(n);
	// subscribers
	_sub_image = it.subscribe(_topic_image_input, 1, &imageEnhance::callback_image_input, this);
	// publishers
	_pub_image = it.advertise(_topic_image_output, 1);
}

imageEnhance::~imageEnhance()
{
}

void imageEnhance::callback_image_input(const sensor_msgs::ImageConstPtr &msg)
{
	// Work on the image.
	try
	{
		cv::Mat frame = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
		cv::Mat image_in;
		cv::Mat image_in_comp;

		if (frame.channels() > 1)
		{
			image_in = frame;
		}
		else
		{
			cv::cvtColor(image_in, frame, cv::COLOR_GRAY2BGR);
		}

		cv::resize(image_in, image_in,
				   cv::Size(image_in.cols /  _scale_factor, image_in.rows / _scale_factor), 0, 0,
				   CV_INTER_LINEAR);

		image_in = cv::Scalar::all(255) - image_in;

		Mat image_out(image_in.rows, image_in.cols, CV_8UC3);
		unsigned char *indata = image_in.data;
		unsigned char *outdata = image_out.data;
		// ROS_INFO("callback_image_in");
		CHazeRemoval hr;
		hr.InitProc(image_in.cols, image_in.rows, image_in.channels());

		if (_use_dehaze)
		{
			ROS_WARN_ONCE("Dehaze");
			hr.Process(indata, outdata, image_in.cols, image_in.rows, image_in.channels());
			// cv::imshow("image_out", image_out);
			// cv::waitKey(0);
		}
		image_out = cv::Scalar::all(255) - image_out;

		sensor_msgs::Image::Ptr output = cv_bridge::CvImage(msg->header, "bgr8", image_out).toImageMsg();
		if (_pub_image.getNumSubscribers() > 0)
			_pub_image.publish(output);
	}
	catch (cv::Exception &e)
	{
		ROS_WARN("Error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
	}
}
