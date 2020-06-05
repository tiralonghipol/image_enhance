#include "ros/ros.h"
#include "image_enhance/image_enhance.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_enhance");
    ros::NodeHandle nh("image_enhance");

    imageEnhance *ie = new imageEnhance(nh, "image_enhance_node", 100);

    ros::spin();
    return 0;
}