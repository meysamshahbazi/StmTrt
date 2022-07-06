#include <opencv2/highgui.hpp>
#include <iostream>

#include "stmtracker.hpp"


using namespace std;
using namespace cv;



int main(int argc, const char ** argv) 
{
    // set input video
    Rect2f roi = Rect(550.0f, 223.0f, 215.0f, 272.0f); // xywh format  
    Mat frame;
    // 000087.jpg
    std::string video {"/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/%06d.jpg"};//= argv[1];
    VideoCapture cap(video);

    // get bounding box
    cap >> frame;
    stmtracker st;
    st.init(frame,roi);
    cap >> frame;
    st.update(frame);
    return 0;
}




