#include "image_enhance/bpdhe.h"

namespace bpdhe
{
    cv::Mat bpdhe::Process(const cv::Mat image_input)
    {
        cv::Mat img = image_input;
        // namedWindow("Input Image", CV_WINDOW_AUTOSIZE);
        //namedWindow("Grayscale", CV_WINDOW_AUTOSIZE);
        // imshow("Input Image", img);

        Mat B = Mat::zeros(1, 256, CV_64F);
        for (int k = 0; k < 255; k++)
        {
            int t1 = 0, t2 = 0, t3 = 0;
            for (int m = 0; m < img.rows; m++)
            {
                for (int n = 0; n < img.cols; n++)
                {
                    if (img.at<cv::Vec3b>(m, n)[0] == k)
                        t1 = t1 + 1;
                    if (img.at<cv::Vec3b>(m, n)[1] == k)
                        t2 = t2 + 1;
                    if (img.at<cv::Vec3b>(m, n)[2] == k)
                        t3 = t3 + 1;
                }
            }

            int t = t1 + t2 + t3;
            double b = 1.0f / (img.rows * img.cols * 3);
            double x = t * b;
            B.at<double>(k + 1) = x;
        }

        Mat T1 = Mat(1, 256, CV_64F, double(0));
        for (int i = 1; i < 256; i++)
        {
            for (int j = 1; j < i; j++)
            {
                T1.at<double>(i) = B.at<double>(j) + T1.at<double>(i);
            }
        }
        Mat T2 = Mat(1, 256, CV_64F, double(0));
        for (int i = 1; i < 256; i++)
        {
            T2.at<double>(i) = round((T1.at<double>(i) * 255) + 0.5);
        }

        Mat bpdhe = img.clone();
        // img.copyTo(bpdhe);
        for (int i = 0; i < 255; i++)
        {
            for (int j = 0; j < img.rows; j++)
            {
                for (int k = 0; k < img.cols; k++)
                {
                    if (bpdhe.at<Vec3b>(j, k)[0] == i)
                        bpdhe.at<Vec3b>(j, k)[0] = T2.at<double>(i + 1);
                    if (bpdhe.at<Vec3b>(j, k)[1] == i)
                        bpdhe.at<Vec3b>(j, k)[1] = T2.at<double>(i + 1);
                    if (bpdhe.at<Vec3b>(j, k)[2] == i)
                        bpdhe.at<Vec3b>(j, k)[2] = T2.at<double>(i + 1);
                }
            }
        }

        imshow("Output Image", B);
        waitKey(1);
        // namedWindow("Ouptut Image", CV_WINDOW_AUTOSIZE);

        // waitKey(0);
        // destroyAllWindows();

        return bpdhe;
    }
} // namespace bpdhe
