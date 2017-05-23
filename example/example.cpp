#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

#include "../include/LBP.h"

int main(int argc, char** argv){

  if(argc != 3){
    std::cout << "Usage:\n"
      "\tTo preview computed LBP image\n"
      "\t./example image_filepath radius(int)\n"
      "\te.g. ./example lena.jpg 3"
      << std::endl;

    return -1;
  }

  const char* filename = argv[1];
  const int radius = std::atoi(argv[2]);
  cv::Mat img = cv::imread(filename, 0);

  cv::Mat LBPimage, ULBPimage;
  LBP::computeLBP(img, LBPimage, radius);
  LBP::computeULBP(img, ULBPimage, radius);

  cv::imshow("original", img);
  cv::imshow("LBP image", LBPimage);
  cv::imshow("ULBP image", ULBPimage);
  cv::waitKey(-1);
}
