#pragma once

#include "common.h"
#include "Pixel.h"

using cv::Mat;
using cv::Range;
using std::list;
using std::vector;

void growImg(const Mat &imgSrc, const Mat &srcMask, const int windowR, Mat &imgDst, Mat &dstMask);
void stichImg(const Mat &imgSrc, const Mat &imgDst, const Mat &dstMask,
              Mat &outputImg);
void getUnfilledNeighbors(const Mat &imgDstMask, list<Pixel> &pixelList);

void getNeighbourhoodWindow(const Pixel &pixel, const Mat &imgDst,
                            const Mat &imgMask, int windowR, Mat &temImg,
                            Mat &maskTemp);
void findMatches(const Mat &imgSrc, const Mat &srcMask, const Mat &temImg, const Mat &maskTemp,
                 const int windowR, vector<Pixel> &bestMatches);

const Pixel &randomPick(const vector<Pixel> &bestMatches);