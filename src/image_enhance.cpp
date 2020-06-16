#include "image_enhance/image_enhance.h"
#include "image_enhance/dehaze.h"
#include "image_enhance/guided_filter.h"
#include "image_enhance/clahe.h"

#define TEST_OPTIMIZATIONS false

imageEnhance::imageEnhance(ros::NodeHandle &n, const std::string &s, int bufSize)
{
	// parameters
	n.getParam("enable_dyn_reconf", _enable_dyn_reconf);

	if (_enable_dyn_reconf)
	{
		_dyn_rec_cb = boost::bind(&imageEnhance::callback_dyn_reconf, this, _1, _2);
		_dr_srv.setCallback(_dyn_rec_cb);
	}

	// ros parameters
	n.getParam("topic_image_input", _topic_image_input);
	n.getParam("topic_image_output", _topic_image_output);
	n.getParam("scale_factor", _scale_factor);
	n.getParam("enable_dehaze", _enable_dehaze);
	// dehaze parameters
	n.getParam("dehaze_radius", m_radius);
	n.getParam("dehaze_omega", m_omega);
	n.getParam("dehaze_t0", m_t0);
	n.getParam("dehaze_r", m_r);
	n.getParam("dehaze_eps", m_eps);
	// clahe parameters
	n.getParam("enable_clahe", _enable_clahe);
	n.getParam("clahe_clip_limit", clahe::_clahe_clip_limit);
	n.getParam("clahe_grid_size", clahe::_clahe_grid_size);

	image_transport::ImageTransport it(n);
	// subscribers
	_sub_image = it.subscribe(_topic_image_input, 1, &imageEnhance::callback_image_input, this);
	// publishers
	m_pub_resized_image = it.advertise(_topic_image_output + "_resized", 1);
	m_pub_dehaze_image = it.advertise(_topic_image_output + "_dehazed", 1);
	m_pub_clahe_image = it.advertise(_topic_image_output + "_clahe", 1);
	#if TEST_OPTIMIZATIONS
		m_pub_dehaze_old_image = it.advertise(_topic_image_output + "_dehazed_old_alg", 1);
	#endif
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
		if (frame.channels() > 1)
		{
			image_in = frame.clone();
		}
		else
		{
			cv::cvtColor(image_in, frame, cv::COLOR_GRAY2BGR);
		}
		cv::resize(image_in, image_in,
					cv::Size(image_in.cols / _scale_factor, image_in.rows / _scale_factor), 
					0, 0, CV_INTER_LINEAR);

		if(m_pub_resized_image.getNumSubscribers() > 0)
		{
			sensor_msgs::Image::Ptr resized_msg = cv_bridge::CvImage(msg->header, "bgr8", image_in).toImageMsg();
			m_pub_resized_image.publish(resized_msg);
		}

		clock_t start = clock();

		Mat image_out(image_in.rows, image_in.cols, CV_8UC3);

		if (_enable_dehaze)
		{

			#if TEST_OPTIMIZATIONS
				Mat image_out_OLD_ALG(image_in.rows, image_in.cols, CV_8UC3);
				ROS_WARN_ONCE("Dehaze Enabled");
				clock_t new_alg_start = clock();
				image_in = cv::Scalar::all(255) - image_in;


				dehaze::CHazeRemoval hr(&image_in, m_omega, m_t0, m_radius, m_r, m_eps);
				hr.Process(image_out);				
				image_out = cv::Scalar::all(255) - image_out;
				ROS_INFO("Dehaze Process Time consumed : %f sec", (float)(clock() - new_alg_start) / CLOCKS_PER_SEC );


				clock_t old_alg_start = clock();

				dehaze::CHazeRemoval hr_OLD_ALG(&image_in, m_omega, m_t0, m_radius, m_r, m_eps);
				hr_OLD_ALG.Process(image_out_OLD_ALG);
				image_out_OLD_ALG = cv::Scalar::all(255) - image_out_OLD_ALG;
				ROS_INFO("Old Dehaze Process Time consumed : %f sec", (float)(clock() - old_alg_start) / CLOCKS_PER_SEC );
			#else
				ROS_WARN_ONCE("Dehaze Enabled");
				clock_t alg_start = clock();
				cv::Mat inverted_image;
				cv::invert(image_in, inverted_image);
				dehaze::CHazeRemoval hr(&inverted_image, m_omega, m_t0, m_radius, m_r, m_eps);
				hr.Process(image_out);
				cv::invert(image_out, image_out);
				ROS_INFO("Dehaze Process Time consumed : %f sec", (float)(clock() - alg_start) / CLOCKS_PER_SEC );
			#endif
				sensor_msgs::Image::Ptr output_msg = cv_bridge::CvImage(msg->header, "bgr8", image_out).toImageMsg();
				if(m_pub_dehaze_image.getNumSubscribers() > 0)
				{
					m_pub_dehaze_image.publish(output_msg);
				}
			#if TEST_OPTIMIZATIONS 
				sensor_msgs::Image::Ptr old_output_msg = cv_bridge::CvImage(msg->header, "bgr8", image_out_OLD_ALG).toImageMsg(); 
				m_pub_dehaze_old_image.publish(old_output_msg);
			#endif
		}
		else
		{
			ROS_WARN_ONCE("Dehaze Disabled");
		}
		if (_enable_clahe)
		{
			ROS_WARN_ONCE("Clahe Enabled");
			clahe::ClaheRos cr;
			image_out = cr.Process(image_in);
			sensor_msgs::Image::Ptr output_msg = cv_bridge::CvImage(msg->header, "bgr8", image_out).toImageMsg();
			m_pub_clahe_image.publish(output_msg);
		}
		else
		{
			ROS_WARN_ONCE("Clahe Disabled");
		}
		clock_t end = clock();
		ROS_DEBUG("Total Enhance Time consumed : %f sec", (float)(clock() - start) / CLOCKS_PER_SEC );
	}
	catch (cv::Exception &e)
	{
		ROS_WARN("Error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
	}
}

void imageEnhance::callback_dyn_reconf(image_enhance::ImageEnhanceConfig &config, uint32_t level)
{
	ROS_INFO("Dynamic Reconfigure Triggered");

	_scale_factor = config.scale_factor;

	_enable_dehaze = config.enable_dehaze;
	m_radius = config.dehaze_radius;
	m_omega = config.dehaze_omega;
	m_t0 = config.dehaze_t0;
	m_r = config.dehaze_r;
	m_eps = config.dehaze_eps;

	_enable_clahe = config.enable_clahe;
	clahe::_clahe_clip_limit = config.clahe_clip_limit;
	clahe::_clahe_grid_size = config.clahe_grid_size;
}
