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
    int m_b_th{ 163 };
    int m_x1{ 30 };
    int m_y1{ 140 };
    int m_x4{ 1470 };
    int m_y4{ 640 };
    int m_lower_bound{ 20 };
    int m_upper_bound{ 2000 };
    int m_dot_area{ 15 };

public:

    

    // Default constructor
    image_defect() {}

    // Initialize a image_defect 
    explicit image_defect(int i) : m_b_th(i), m_x1(i), m_y1(i), m_x4(i), 
        m_y4(i), m_lower_bound(i), m_upper_bound(i), m_dot_area(i)
    {}

   
    image_defect(int b_th, int x1, int y1, int x4, int y4, int lower_bound,
        int upper_bound, int dot_area)
        : m_b_th(b_th), m_x1(x1), m_y1(y1), m_x4(x4), m_y4(y4)
        , m_lower_bound(lower_bound), m_upper_bound(upper_bound)
        , m_dot_area(dot_area)
    {}

    int r_lower_bound() { return m_lower_bound; }
    int r_b_th() { return m_b_th; }
    int r_x1() { return m_x1; }
    int r_y1() { return m_y1; }
    int r_x4() { return m_x4; }
    int r_y4() { return m_y4; }
    

    std::string name;
    void setName(std::string);
};




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


int main(int argc, const char** argv)
{
    //Read
    cv::Mat img = cv::imread("C:\\Users\\wade.huang\\Desktop\\code\\CCD-Test-master\\data\\LAAK256002_NG_A_431_1.bmp");

    image_defect D ;
    const int s1   = D.r_lower_bound();
    const int b_th = D.r_b_th();
    const int x1   = D.r_x1();
    const int y1   = D.r_y1();
    const int x4   = D.r_x4();
    const int y4   = D.r_y4();

    cv::Mat blur_img;
    cv::GaussianBlur(img, blur_img, Size(3,3),0,0);
     
    cv::Mat gray_img;
    cv::cvtColor(blur_img, gray_img, cv::COLOR_BGR2GRAY);

    cv::Mat image_final;
    cv::adaptiveThreshold(gray_img, image_final, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 13, 2);
    
    
    vector<vector<Point> > contours0;
    vector<Vec4i> hierarchy;
    cv::findContours(image_final, contours0, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

  
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
            contourArea(contours0[i]) > s1
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
