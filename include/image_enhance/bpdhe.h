#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

namespace bpdhe
{
    class bpdhe
    {
    private:
        /* data */
    public:
        bpdhe(/* args */);
        ~bpdhe();
        cv::Mat Process(const cv::Mat image_input);
    };

    bpdhe::bpdhe(/* args */)
    {
    }

    bpdhe::~bpdhe()
    {
    }

} // namespace bpdhe