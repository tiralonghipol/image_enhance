#pragma once

#include "ros/ros.h"
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dynamic_reconfigure/server.h>
#include <image_enhance/ImageEnhanceConfig.h>

using namespace std;
using namespace cv;

class imageEnhance
{

public:
    imageEnhance(ros::NodeHandle &n, const std::string &s, int bufSize);
    ~imageEnhance();
    void callback_image_input(const sensor_msgs::ImageConstPtr &msg);
    void callback_dyn_reconf(image_enhance::ImageEnhanceConfig &config, uint32_t level);
    // dynamic reconfigure
    dynamic_reconfigure::Server<image_enhance::ImageEnhanceConfig> _dr_srv;
    dynamic_reconfigure::Server<image_enhance::ImageEnhanceConfig>::CallbackType _dyn_rec_cb;

private:
    // publishers
    image_transport::Publisher _pub_image;
    image_transport::Publisher _old_pub_image;
    // subscribers
    image_transport::Subscriber _sub_image;

    // parameters
    string _topic_image_input;
    string _topic_image_output;
    bool _enable_dyn_reconf;
    int _scale_factor;
    bool _enable_dehaze;
    bool _enable_clahe;

    
};
