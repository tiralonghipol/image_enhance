#pragma once

#include "ros/ros.h"
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <numeric>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class imageEnhance
{

public:
    imageEnhance(ros::NodeHandle &n, const std::string &s, int bufSize);
    ~imageEnhance();
    void callback_image_input(const sensor_msgs::ImageConstPtr &msg);

private:
    // publishers
    image_transport::Publisher _pub_image;
    // subscribers
    image_transport::Subscriber _sub_image;

    // parameters
    string _topic_image_input;
    string _topic_image_output;
    bool _enable_dyn_reconf;
    bool _use_dehaze;
    int _scale_factor;
};
