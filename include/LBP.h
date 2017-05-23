#ifndef _LBP_H_
#define _LBP_H_

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <array>


namespace LBP {

    // code locations with indecies like
    // 0 1 2
    // 7 * 3
    // 6 5 4
    const std::array<int, 8>
        CODE = {0, 1, 2, 3, 4, 5, 6, 7};

    const cv::Mat 
        ULBP_LUT = (
        cv::Mat_<uint8_t>(1, 256) << 
            1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0,
            13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0,
            0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0,
            0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0,
            0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58
        );

    void computeLBP(cv::InputArray image_, cv::OutputArray dst, int radius=1){
        cv::Mat image = image_.getMat();
        cv::Mat src;
        
        // convert to gray-scaled image
        if(image.channels() == 3){
            cv::cvtColor(image, src, CV_BGR2GRAY);
        } else if(image.channels() == 1){
            image.copyTo(src);
        } else {
            throw "input image must have 1 or 3 channels";
        }

        const int width = src.cols;
        const int height = src.rows;

        cv::Mat lbpimage = cv::Mat::zeros(
                height - 2*radius,
                width - 2*radius,
                CV_8UC1
            );


        for(int y = radius; y < height-radius; ++y){
            for(int x = radius; x < width-radius; ++x){
                const std::uint8_t ref = src.data[y * width + x];

                std::uint8_t value = 0;
                value += (src.data[(y-radius)*width + (x-radius)] > ref)
                    << (7 - CODE[0]);
                value += (src.data[(y-radius)*width + (x)]        > ref)
                    << (7 - CODE[1]);
                value += (src.data[(y-radius)*width + (x+radius)] > ref)
                    << (7 - CODE[2]);
                value += (src.data[(y)*width        + (x+radius)] > ref)
                    << (7 - CODE[3]);
                value += (src.data[(y+radius)*width + (x+radius)] > ref)
                    << (7 - CODE[4]);
                value += (src.data[(y+radius)*width + (x)]        > ref)
                    << (7 - CODE[5]);
                value += (src.data[(y+radius)*width + (x-radius)] > ref)
                    << (7 - CODE[6]);

                lbpimage.data[lbpimage.cols * (y - radius) + (x - radius)] = value;
            }
        }

        lbpimage.copyTo(dst);
    }

    void computeULBP(cv::InputArray image_, cv::OutputArray dst, int radius=1){

        cv::Mat image = image_.getMat();
        cv::Mat src;
        
        // convert to gray-scaled image
        if(image.channels() == 3){
            cv::cvtColor(image, src, CV_BGR2GRAY);
        } else if(image.channels() == 1){
            image.copyTo(src);
        } else {
            throw "input image must have 1 or 3 channels";
        }

        cv::Mat lbp, ulbp;
        computeLBP(src, lbp, radius);

        cv::LUT(lbp, ULBP_LUT, ulbp);

        ulbp.copyTo(dst);
    }
}

#endif // _LBP_H_
