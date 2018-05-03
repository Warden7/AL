#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <experimental/filesystem>
#include <math.h>

namespace fs = std::experimental::filesystem;

double contrast_measure( const cv::Mat& src)
{
    cv::Mat Gx, Gy;
    cv::Sobel( src, Gx, CV_32F, 1, 0 );
    cv::Sobel( src, Gy, CV_32F, 0, 1 );
    double normGx = cv::norm( Gx );
    double normGy = cv::norm( Gy );
    double sumSq = normGx * normGx + normGy * normGy;
    return static_cast<float>( 1. / ( sumSq / src.size().area() + 1e-6 ));
}

float blurEvaluate(cv::Mat& img)
{
    //check if image exists
    if(!img.data)
    {
        std::cout << "Image may not exist." << std::endl;
        return -1;
    }
    // use Laplacian calculation
    cv::Mat image;
    cv::Laplacian(img, image, CV_32F, 3);

    // display image
    //cv::imshow( "Display window", image);
    //cv::waitKey(0);
     
    cv::Mat var, tmp;
    cv::meanStdDev(image, tmp, var);

    //std::cout << "The rows=" << var.rows << ", the cols=" << var.cols << std::endl;
    double stddev;
    double variance = 0;
    for(int i = 0; i < var.rows; i++)
    {
        stddev = var.at<double>(i, 0);
        variance += stddev * stddev;
        //std::cout << "i=" << i << "=" << stddev << std::endl;
    }

    return variance/3.0;
}

float getImageMeanBrightness(cv::Mat& img)
{
    cv::Mat image_gray, image_float;
    cv::cvtColor(img, image_gray, CV_RGB2GRAY);
    image_gray.convertTo(image_float, CV_32F);
    cv::Scalar scalar = cv::mean(image_float);

    return scalar.val[0];
}

cv::Mat imageResize(cv::Mat& img, float scale_thred)
{
    cv::Mat img_resize;
    float ratio;
    int rows,cols,len_max;
    int rows_target,cols_target;

    rows = img.rows;
    cols = img.cols;
    len_max = (rows >= cols) ? rows : cols;

    //std::cout << "rows:" <<rows << ", " <<" cols :"<< cols << '\n';
    //std::cout << "len_max:" <<len_max << '\n';
    if(len_max > scale_thred)
    {
        if(rows <= cols)
        {
            ratio = scale_thred/cols;
            cols_target = (int)(scale_thred);
            rows_target = (int)(ratio*rows);
        }else
        {
            ratio = scale_thred/rows;
            cols_target = (int)(ratio*cols);
            rows_target = (int)(scale_thred);
        }
        cv::resize(img, img_resize, cv::Size(cols_target, rows_target), CV_INTER_LINEAR);
        //std::cout << "rows:" <<rows << ", " <<" cols :"<< cols << " rows_target :"<< rows_target << " cols_target:" <<cols_target << ", " << '\n';

    }else
    {
        img_resize = img;
    }

    return img_resize;

}

int main(int argc, char** argv)
{
    cv::Mat img, img_resize;

    float mean,blur;
    float mean_thred = 35;
    float blur_thred = 450;
    float scale_thred = 640.0;

    for(auto& p: fs::directory_iterator(argv[1])) {
        
        img = cv::imread(p.path().string());
        img_resize = imageResize(img, scale_thred);
        mean = getImageMeanBrightness(img_resize);
        blur = blurEvaluate(img_resize);

        //std::cout << "rows:" <<img.rows << ", " <<" cols :"<< img.cols << "rows_target:" <<img_resize.rows << ", " <<" cols_target :"<< img_resize.cols << '\n';

        if(mean < mean_thred || blur < blur_thred) {
            std::cout << "[BLUR]:" <<p << ", " <<" blur :"<< blur << "  mean :"<< mean << '\n';

        }else {
            std::cout << "[CLEAR]:" <<p << ", " <<" blur :"<< blur << "  mean :"<< mean << '\n';
        }
    }

    img.release();
}
