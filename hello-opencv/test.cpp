// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// https://stackoverflow.com/questions/23506105/extracting-text-opencv?noredirect=1&lq=1

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;


class image_defect
{
private:

public:
    image_defect();
    std::string name;
    void setName(std::string);
};

// define the class constructor
image_defect::image_defect()
{
    //brightness
    int b_th = 163;
    int x1 = 30;
    int y1 = 140;
    int x4 = 1470;
    int y4 = 640;
    // 光影誤判面積大約為10 pixel
    // 瑕疵面積大小
    int lower_bound = 20;
    int upper_bound = 2000; 
    int dot_area = 15;
}

void image_defect::setName(std::string excel_name)
{
    name = excel_name;
}




std::vector<cv::Rect> detectLetters(cv::Mat img)
{
    std::vector<cv::Rect> boundRect;
    cv::Mat img_gray, img_sobel, img_threshold, element;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    cv::Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::threshold(img_sobel, img_threshold, 0, 255, THRESH_OTSU + THRESH_BINARY);
    
    cv::imwrite("img_sobel.jpg", img_sobel);
    cv::imwrite("img_threshold.jpg", img_threshold);
    
    
    element = getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
    cv::morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element); //Does the trick
    
    cv::imwrite("img_threshold_2.jpg", img_threshold);
    
    std::vector< std::vector< cv::Point> > contours;
    cv::findContours(img_threshold, contours, 0, 1);
    std::vector<std::vector<cv::Point> > contours_poly(contours.size());
    for (int i = 0; i < contours.size(); i++)
        if (contours[i].size() > 100)
        {
            cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
            cv::Rect appRect(boundingRect(cv::Mat(contours_poly[i])));
            if (appRect.width > appRect.height)
                boundRect.push_back(appRect);
        }
    return boundRect;
}

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
    double i = fabs(contourArea(cv::Mat(contour1)));
    double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}

int main(int argc, const char** argv)
{

    //Read
    cv::Mat img = cv::imread("C:\\Users\\wade.huang\\Desktop\\code\\CCD-Test-master\\data\\LAAK256002_NG_A_431_1.bmp");

    int s1 = 5;
    int s2 = 4000;
    cv::Mat blur_img;
    cv::GaussianBlur(img, blur_img, Size(3,3),0,0);
     
    cv::Mat gray_img;
    cv::cvtColor(blur_img, gray_img, cv::COLOR_BGR2GRAY);

    cv::Mat image_final;
    cv::adaptiveThreshold(gray_img, image_final, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 13, 2);
    
    
    vector<vector<Point> > contours0;
    vector<Vec4i> hierarchy;
    cv::findContours(image_final, contours0, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // comparison function object
 
    
    /*
    std::sort(contours0.begin(), contours0.end(), compareContourAreas);
    // grab contours
    std::vector<cv::Point> biggestContour = contours0[contours0.size() - 1];
    std::vector<cv::Point> smallestContour = contours0[0];
    */
    int b_th = 163;
    const int x1 = 80;
    const int y1 = 140;
    const int x4 = 1600;
    const int y4 = 640;

    
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x4, y4), 
        (0, 255, 255), 2);

    vector<Moments> mu(contours0.size());

    for (int i = 0; i < contours0.size(); i++)
    {
        mu[i] = cv::moments(contours0[i]);
    }
    
    std::cout << "x1 " << x1;
    std::cout << "y1 " << y1;

    vector<Point2f> mc(contours0.size());
    for (size_t i = 0; i < contours0.size(); i++)
    {
        
        //add 1e-5 to avoid division by zero
        /*
        mc[i] = Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
            static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
        cout << "mc[" << i << "]=" << mc[i] << endl;
        */

        int x = int(mu[i].m10 / (mu[i].m00 + 1e-5));
        int y = int(mu[i].m01 / (mu[i].m00 + 1e-5));

        
        

        if (x4 > x && x > x1 && y4 > y && y > y1 && 
            contourArea(contours0[i]) > 10 
            && (int)gray_img.at<uchar>(y, x) > 80
           )
        {
            
            cv::circle(img, cv::Point(x, y), 5, (255, 255, 30), 3);
        }
        //cv::circle(img, cv::Point(x, y), 3, (36, 255, 12), 1);
        
    }
    
    
    cout << (int)gray_img.at<uchar>(2, 2);
    cv::circle(img, cv::Point(x1, y1), 3, (255, 255, 30), 3);
    
    cv::imshow("Gray scale", gray_img);
    cv::imshow("Binary", image_final);
    cv::imshow("result", img);

    cv::waitKey(0);

    return 0;
}