/**
 * @file growImg.cpp
 * @author Hawk Wang (yourwanghao@gmail.com)
 * @brief
 *  This is a simple naive implementation of
 *  Efros and Leung, “Texture Synthesis by Non-parametric Sampling,” ICCV1999
 *
 *  It is slow and experimental, you risk yourself by using it.
 *
 * @version 1.0
 * @date 2019-06-07
 *
 * @copyright Copyright (c) 2019
 */

/**
Copyright 2019 Hawk Wang

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <strstream>
#include <omp.h>

#include "Pixel.h"
#include "common.h"
#include "growImg.h"
#include "testGrowImg.h"

using cv::Mat;
using cv::Range;
using std::list;
using std::vector;

void saveInternalImg(const Mat &imgDst, const Mat &dstMask, const int iter) {
      std::ostrstream dstMaskName;  // dynamic buffer
      dstMaskName << "./imgs/dstMask" << iter << ".jpg" << std::ends;
      Mat dstMaskSave = dstMask.clone();
      dstMaskSave *= 255;
      cv::imwrite(dstMaskName.str(), dstMaskSave);

      std::ostrstream imgDstName;
      imgDstName << "./imgs/imgDst" << iter << ".jpg" << std::ends;
      cv::imwrite(imgDstName.str(), imgDst);
  }

/**
 * @brief
 *
 * @param imgSrc
 * @param windowR
 * @param imgDst
 * @param dstMask
 */
void growImg(const Mat &imgSrc, const Mat &srcMask, const int windowR, Mat &imgDst, Mat &dstMask) {
  float MaxErrThreshold = 0.3;
  int iter = 0;
  int totalPixelCount = dstMask.rows * dstMask.cols;

  while (cv::countNonZero(dstMask) != totalPixelCount) {
    // printf("cv::sum(dstMask)[0]=%d, totalPixelCount=%d\n",
    //        cv::countNonZero(dstMask), totalPixelCount);
    printf("iter=%d\n", iter);

    saveInternalImg(imgDst, dstMask, iter);

    int progress = 0;
    list<Pixel> pixelList;
    getUnfilledNeighbors(dstMask, pixelList);

    // std::cout << "left pixel count=" << pixelList.size() << std::endl;
    // std::cout << "MaxErrThreshold=" << MaxErrThreshold << std::endl;
    int count = 0;
    for (Pixel pixel : pixelList) {
      //   pixel.show();
      Mat temImg;
      Mat maskTemp;
      getNeighbourhoodWindow(pixel, imgDst, dstMask, windowR, temImg, maskTemp);
      vector<Pixel> bestMatches;
      findMatches(imgSrc, srcMask, temImg, maskTemp, windowR, bestMatches);

      //   printf("%d/%d, bestMatches(%d)\n", count, pixelList.size(),
      //          bestMatches.size());

      Pixel bestMatch = randomPick(bestMatches);
      //   std::cout << "bestMatch.error" << bestMatch.error << std::endl;

      if (bestMatch.error < MaxErrThreshold) {
        imgDst.at<uchar>(pixel.y, pixel.x) = bestMatch.value;
        dstMask.at<uchar>(pixel.y, pixel.x) = 1;
        progress = 1;
      }
      count++;
    }
    if (progress == 0) {
      MaxErrThreshold *= 1.1;
    }

    iter++;
    // break;
  }
  printf("End of %s\n\r", __FUNCTION__);
  saveInternalImg(imgDst, dstMask, iter);
}

/**
 * @brief Get the Unfilled Neighbors object
 *
 * returns a list of all unfilled pixels that have filled pixels as their
 * neighbors. The image is subtracted from its morphological dilation). The list
 * is randomly permuted and then sorted by decreasing number of filled neighbor
 * pixels.
 *
 * @param dstMask
 * @param pixelList
 */
void getUnfilledNeighbors(const Mat &dstMask, list<Pixel> &pixelList) {
  Mat kernel = Mat::ones(3, 3, CV_8UC1);
  kernel.at<uchar>(1, 1) = 0;
  Mat imgNeighbourCount;
  cv::filter2D(dstMask, imgNeighbourCount, -1, kernel);

  Mat newmask = dstMask.clone();
  dilate(dstMask, newmask, Mat());
  Mat roi = newmask - dstMask;

  imgNeighbourCount.setTo(0, 1 - roi);

  //   std::cout << imgNeighbourCount.size() << std::endl;

  for (int i = 0; i < imgNeighbourCount.rows; i++) {
    for (int j = 0; j < imgNeighbourCount.cols; j++) {
      if (imgNeighbourCount.at<uchar>(i, j) != 0) {
        Pixel p(i, j, imgNeighbourCount.at<uchar>(i, j));
        pixelList.push_back(p);
      }
    }
  }

  //   vector<Pixel> tmp(pixelList);
  //   std::random_device rd;
  //   std::mt19937 g(rd());
  //   std::shuffle(pixelList.begin(), pixelList.end(), g);
  pixelList.sort(cmp);
}

/**
 * @brief Get the Neighbourhood Window Around a Given Pixel
 *
 * @param pixel
 * @param imgDst
 * @param imgMask
 * @param windowR
 * @param temImg
 * @param maskTemp
 */
void getNeighbourhoodWindow(const Pixel &pixel, const Mat &imgDst,
                            const Mat &dstMask, const int windowR, Mat &imgTemp,
                            Mat &maskTemp) {
  // make border to prevent handling the edge
  Mat imgDstB, dstMaskB;
  cv::copyMakeBorder(imgDst, imgDstB, windowR, windowR, windowR, windowR,
                     cv::BORDER_CONSTANT, 0);
  cv::copyMakeBorder(dstMask, dstMaskB, windowR, windowR, windowR, windowR,
                     cv::BORDER_CONSTANT, 0);

  int y = pixel.y;
  int x = pixel.x;

  imgTemp =
      imgDstB(Range(y, y + 2 * windowR + 1), Range(x, x + 2 * windowR + 1));
  maskTemp =
      dstMaskB(Range(y, y + 2 * windowR + 1), Range(x, x + 2 * windowR + 1));

  return;
}

cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay) {
  auto gauss_x = cv::getGaussianKernel(cols, sigmax, CV_32F);
  auto gauss_y = cv::getGaussianKernel(rows, sigmay, CV_32F);
  return gauss_x * gauss_y.t();
}

/**
 * @brief Find patches that match the input template in the source image
 *
 * @param imgSrc
 * @param temImg
 * @param maskTemp
 * @param bestMatches
 */
void findMatches(const Mat &imgSrc, const Mat &srcMask, const Mat &imgTemp, const Mat &maskTemp,
                 const int windowR, vector<Pixel> &bestMatches) {
  // make border to prevent handling the edge
  Mat imgSrcB, srcMaskB;
  cv::copyMakeBorder(imgSrc, imgSrcB, windowR, windowR, windowR, windowR,
                     cv::BORDER_CONSTANT, 0);
  cv::copyMakeBorder(srcMask, srcMaskB, windowR, windowR, windowR, windowR,
                     cv::BORDER_CONSTANT, 0);
                     

  assert((imgTemp.cols == 2 * windowR + 1) &&
         (imgTemp.rows == 2 * windowR + 1));

  const Mat validMask = maskTemp;
  int winSize = 2 * windowR + 1;
  float sigma = winSize / 6.4;

  Mat gaussMask = getGaussianKernel(winSize, winSize, sigma, sigma);
  gaussMask.setTo(0, 1 - validMask);
  float totalWeight = sum(gaussMask)[0] + 1e-9;
  //   std::cout << "gaussMask Center " << gaussMask.at<float>(windowR, windowR)
  //             << std::endl;
  //   std::cout << "gaussMask Origin" << gaussMask.at<float>(0, 0) <<
  //   std::endl; std::cout << " total count =" << totalWeight << " valid mask
  //   sum="<<sum(validMask)[0] <<std::endl;

  Mat imgSSD = Mat::zeros(imgSrcB.size(), CV_32FC1);
  float minDist = 1e+9;
  float errThreshold = 0.1;

  #pragma omp parallel for
  for (int i = windowR; i < imgSrcB.rows - windowR; i++) {
    float *imgSSDLine = imgSSD.ptr<float>(i);
    const uchar *srcMaskBLine = srcMaskB.ptr<uchar>(i);
    for (int j = windowR; j < imgSrcB.cols - windowR; j++) {
      if (srcMaskBLine[j]==0) {
          continue;
      }
      const Mat roiSrc = imgSrcB(Range(i - windowR, i + windowR + 1),
                                 Range(j - windowR, j + windowR + 1));
      assert(roiSrc.size() == imgTemp.size());
      for (int ii = 0; ii < imgTemp.rows; ii++) {
        for (int jj = 0; jj < imgTemp.cols; jj++) {
          float dist = (imgTemp.at<uchar>(ii, jj) / 255.0f -
                        roiSrc.at<uchar>(ii, jj) / 255.0f);
          dist = dist * dist * gaussMask.at<float>(ii, jj);
          imgSSDLine[j] += dist;
        }
        // imgSSDLine[j - windowR] /= totalWeight;
      }
      if (imgSSDLine[j] < minDist) {
        minDist = imgSSDLine[j];
      }
    }
  }
  imgSSD = imgSSD(Range(windowR, imgSSD.rows - windowR),
                  Range(windowR, imgSSD.cols - windowR));
  imgSSD.setTo(1e+9, srcMask==0);

  //   std::cout << "minDist=" << minDist << std::endl;
  double minV, maxV;
  cv::minMaxLoc(imgSSD, &minV, &maxV);
  //   std::cout << "imgSSD min: " << minV << " max:" << maxV << std::endl;

  for (int i = 0; i < imgSSD.rows; i++) {
    float *imgSSDLine = imgSSD.ptr<float>(i);
    const uchar *imgSrcLine = imgSrc.ptr<uchar>(i);
    for (int j = 0; j < imgSSD.cols; j++) {
      if (imgSSDLine[j] <= minDist * (1 + errThreshold)) {
        Pixel p(i, j, 0);
        p.value = imgSrcLine[j];
        p.error = imgSSDLine[j];

        bestMatches.push_back(p);
      }
    }
  }
  //   std::cout << "length of bestMatches=" << bestMatches.size() << std::endl;

  return;
}

const Pixel &randomPick(const vector<Pixel> &bestMatches) {
  int pixelNumber = bestMatches.size();
  int random_number = std::rand();
  int index = (float)random_number / RAND_MAX * pixelNumber;
  return bestMatches[index];
}