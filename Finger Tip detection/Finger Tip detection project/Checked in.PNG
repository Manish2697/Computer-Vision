#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const int max_value_H = 180;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 21, low_V = 44;
int high_H = max_value_H, high_S = 255, high_V = max_value;
static void on_low_H_thresh_trackbar(int, void*)
{
    low_H = min(high_H - 1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
    high_H = max(high_H, low_H + 1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
    low_S = min(high_S - 1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
    high_S = max(high_S, low_S + 1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
    low_V = min(high_V - 1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
    high_V = max(high_V, low_V + 1);
    setTrackbarPos("High V", window_detection_name, high_V);
}


int track_rows = 0;
int nContours = 0;

/*-----------------------PALM SEPERATION----------------------------*/

Mat palm_separation(Mat input) {
    Mat input_image, current_image, output_image;
    int nRows, nCols, current_image_size, next_image_size, loop;
    uchar* image_ptr;
    vector <vector <Point> > contours;
    vector<Vec4i> hierarchy;

    input_image = input.clone();
    output_image = input.clone();

    /*-----------------------Intialzing variables------------------*/
    nRows = loop = input_image.rows;
    nCols = input_image.cols;
    current_image_size = input_image.total();
    next_image_size = 0;
    image_ptr = input_image.ptr< uchar >(0);

    /*---To find Number of Contours-----*/
    while (loop > 0) {
        next_image_size = current_image_size - nCols;

        for (int i = current_image_size; i > next_image_size; i--) {
            image_ptr[i] = 0;
        }

        current_image = input_image.clone();

        int contours_count = 0;

        findContours(current_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));


        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() > 30) {
                contours_count += 1;

                if (contours_count > nContours) {
                    nContours = contours_count;
                    track_rows = loop;
                }
            }
        }

        current_image_size = next_image_size;
        loop--;
    }

    uchar* image_ptr_new = output_image.ptr < uchar >(0);
    loop = nRows;
    int current_image_size_new = output_image.total();


    while (loop >= track_rows) {
        next_image_size = current_image_size_new - nCols;
        for (int i = current_image_size_new; i > next_image_size; i--) {
            image_ptr_new[i] = 0;
        }
        current_image_size_new = next_image_size;
        loop--;
    }
    return output_image;
}

int main(int argc, char** argv)
{
    VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);

    namedWindow(window_capture_name);
    namedWindow(window_detection_name);

    // Trackbars to set thresholds for HSV values
    createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);

    Mat frame, frame_HSV, frame_threshold, gray1;
    while (1) {

        while (true) {
            cap >> frame;

            if (frame.empty())
            {
                break;
            }

            /*Convert image from bgr to hsv*/
            cvtColor(frame, frame_HSV, COLOR_BGR2HSV);

            /* Object detection using different HSV values */
            inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);


            /* Palm separation function to separate palm from fingers */
            Mat img1 = palm_separation(frame_threshold);

            Mat canny_output;
            int thresh = 100;
            Canny(frame_HSV, canny_output, thresh, thresh * 2, 3);
            
            Mat dst = Mat::zeros(img1.rows, img1.cols, CV_8UC3);
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;

            findContours(img1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));


            vector<RotatedRect> minRect(contours.size());
            float scale = 0.34;

            for (int i = 0; i < contours.size(); i++) {
                minRect[i] = minAreaRect(Mat(contours[i]));
            }

            for (int i = 0; i < contours.size(); i++) {
                if (contours[i].size() > 70) {

                    Point2f rect_points[4];
                    minRect[i].points(rect_points);

                    if (minRect[i].size.height > minRect[i].size.width) {
                        minRect[i].center = (rect_points[1] + rect_points[2]) / 2 + (rect_points[0] - rect_points[1]) / 6;
                        minRect[i].size.height = (float)scale * (minRect[i].size.height);
                    }
                    else {
                        minRect[i].center = (rect_points[2] + rect_points[3]) / 2 + (rect_points[0] - rect_points[3]) / 6;
                        minRect[i].size.width = (float)scale * (minRect[i].size.width);
                    }

                    minRect[i].points(rect_points);
                    for (int j = 0; j < 4; j++)
                        line(frame, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
                }
            }

            imshow("video", frame);

            char key = (char)waitKey(10);
            if (key == 'c')
            {
                break;
            }
        }
    }
    return 0;
}