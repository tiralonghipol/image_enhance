#include "image_enhance/image_enhance.h"
#include "image_enhance/dehaze.h"
#include "image_enhance/guided_filter.h"

imageEnhance::imageEnhance(ros::NodeHandle &n, const std::string &s, int bufSize)
{
	// parameters
	n.getParam("enable_dyn_reconf", _enable_dyn_reconf);

	if (_enable_dyn_reconf)
	{
		// dynamic reconfigure
		_dyn_rec_cb = boost::bind(&imageEnhance::callback_dyn_reconf, this, _1, _2);
		_dr_srv.setCallback(_dyn_rec_cb);
	}

	// ros parameters
	n.getParam("topic_image_input", _topic_image_input);
	n.getParam("topic_image_output", _topic_image_output);
	n.getParam("use_dehaze", _use_dehaze);
	n.getParam("scale_factor", _scale_factor);
	// filter paramters
	n.getParam("dehaze_radius", dehaze::_radius);
	n.getParam("dehaze_omega", dehaze::_omega);
	n.getParam("dehaze_t0", dehaze::_t0);
	n.getParam("dehaze_r", dehaze::_r);
	n.getParam("dehaze_eps", dehaze::_eps);

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
				   cv::Size(image_in.cols / _scale_factor, image_in.rows / _scale_factor), 0, 0,
				   CV_INTER_LINEAR);

		clock_t start = clock();
		image_in = cv::Scalar::all(255) - image_in;

		Mat image_out(image_in.rows, image_in.cols, CV_8UC3);
		unsigned char *indata = image_in.data;
		unsigned char *outdata = image_out.data;
		// ROS_INFO("callback_image_in");
		dehaze::CHazeRemoval hr;
		hr.InitProc(image_in.cols, image_in.rows, image_in.channels());

		if (_use_dehaze)
		{
			ROS_WARN_ONCE("Dehazing Activated");
			hr.Process(indata, outdata, image_in.cols, image_in.rows, image_in.channels());
		}
		image_out = cv::Scalar::all(255) - image_out;

		clock_t end = clock();
		cout << "Time consumed : " << (float)(end - start) / CLOCKS_PER_SEC << "s" << endl;

		sensor_msgs::Image::Ptr output = cv_bridge::CvImage(msg->header, "bgr8", image_out).toImageMsg();
		if (_pub_image.getNumSubscribers() > 0)
			_pub_image.publish(output);
	}
	catch (cv::Exception &e)
	{
		ROS_WARN("Error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
	}
}

void imageEnhance::callback_dyn_reconf(image_enhance::ImageEnhanceConfig &config, uint32_t level)
{
	ROS_WARN_ONCE("Dynamic Reconfigure Triggered");

	_scale_factor = config.scale_factor;
	dehaze::_radius = config.dehaze_radius;
	dehaze::_omega = config.dehaze_omega;
	dehaze::_t0 = config.dehaze_t0;
	dehaze::_r = config.dehaze_r;
	dehaze::_eps = config.dehaze_eps;
	// ROS_INFO("dehaze_radius = %d", dehaze::_radius);
}
