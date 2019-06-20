#include <iostream>
#include <list>
#include <opencv2/opencv.hpp>
#include <vector>

using cv::Mat;
using cv::Point;
using cv::Range;
using cv::RNG;
using cv::Size;
using cv::Vec3b;
using std::list;
using std::vector;

static const int COUNT_SSD = 3;

static RNG rng(0);

int computeSSD(const Mat &img1, const Mat &img2) {
  assert(img1.size() == img2.size());
  int h = img1.rows;
  int w = img1.cols;
  int sum = 0;

  for (int y = 0; y < h; y++) {
    const Vec3b *p1 = img1.ptr<Vec3b>(y);
    const Vec3b *p2 = img2.ptr<Vec3b>(y);
    for (int x = 0; x < w; x++) {
      for (int c = 0; c < 3; c++) {
        int psum = (int)p1[x][c] - (int)p2[x][c];
        sum += psum * psum;
      }
    }
  }

  return sum;
}

bool cmp(const vector<int> &a, const vector<int> &b) {
  if (a[2] < b[2]) {
    return true;
  } else {
    return false;
  }
}

Point searchSrcPatch(const Mat srcImg, const Mat dstOverlapL,
                     const Mat dstOverlapT, int edgeLen, int overlap) {
  int srcH = srcImg.rows;
  int srcW = srcImg.cols;

  int curYY;
  int curXX;
  int minssd = 1e+8;

  list<vector<int>> keypoints;
  for (int yy = 0; yy < srcH - edgeLen; yy = yy + overlap) {
    for (int xx = 0; xx < srcW - edgeLen; xx = xx + overlap) {
      int ssd = 0;
      if (!dstOverlapL.empty()) {
        Mat srcOverlapL =
            srcImg(Range(yy, yy + edgeLen), Range(xx, xx + overlap));

        ssd += computeSSD(dstOverlapL, srcOverlapL);
      }

      if (!dstOverlapT.empty()) {
        Mat srcOverlapT =
            srcImg(Range(yy, yy + overlap), Range(xx, xx + edgeLen));
        ssd += computeSSD(dstOverlapT, srcOverlapT);
      }

      if (ssd < minssd) {
        minssd = ssd;
        curYY = yy;
        curXX = xx;
      }

      vector kp = {curYY, curXX, ssd};
      keypoints.push_back(kp);
    }
  }

  keypoints.sort(cmp);
  int pos = rng.uniform(0, COUNT_SSD);
  int cpos = 0;
  for (auto kp : keypoints) {
    if (cpos != pos) {
      cpos++;
      continue;
    } else {
      curYY = kp[0];
      curXX = kp[1];
      printf("pos=%d, ssd=%d, minssd=%d\n",pos, kp[2], minssd);
      break;
    }
  }

  Point ret(curXX, curYY);
  return ret;
}

void synthesis(const Mat &srcImg, const int edgeLen, const int overlap,
               const Size &targetSize, Mat &dstImg) {
  // Step0. Basic input check
  int srcH = srcImg.rows;
  int srcW = srcImg.cols;

  assert((edgeLen <= srcH) && (edgeLen <= srcW));
  assert((srcH < targetSize.height) && (srcW < targetSize.width));

  if (dstImg.empty() || dstImg.size() != targetSize) {
    dstImg = Mat::zeros(targetSize, srcImg.type());
  }
  int dstH = dstImg.rows;
  int dstW = dstImg.cols;
  for (int y = 0; y < dstH - edgeLen; y = y + edgeLen - overlap) {
    for (int x = 0; x < dstW - edgeLen; x = x + edgeLen - overlap) {
      Mat patch;
      if ((y == 0) && (x == 0)) {
        // choose a random patch from source image
        int patchY = rng.uniform(0, srcH - edgeLen);
        int patchX = rng.uniform(0, srcW - edgeLen);
        patch = srcImg(Range(patchY, patchY + edgeLen),
                       Range(patchX, patchX + edgeLen));
        assert(patch.size() == Size(edgeLen, edgeLen));
      } else {
        Point srcPoint;
        Mat dstOverlapL;
        Mat dstOverlapT;
        if (y == 0) {
          // constraint by left overlap pixels
          dstOverlapL = dstImg(Range(y, y + edgeLen), Range(x, x + overlap));
        } else if (x == 0) {
          // constraint by left overlap pixels
          dstOverlapT = dstImg(Range(y, y + overlap), Range(x, x + edgeLen));
        } else {
          // constraint by both left and top overlap pixels
          dstOverlapL = dstImg(Range(y, y + edgeLen), Range(x, x + overlap));
          Mat dstOverlapT =
              dstImg(Range(y, y + overlap), Range(x, x + edgeLen));
        }
        srcPoint =
            searchSrcPatch(srcImg, dstOverlapL, dstOverlapT, edgeLen, overlap);
        patch = srcImg(Range(srcPoint.y, srcPoint.y + edgeLen),
                       Range(srcPoint.x, srcPoint.x + edgeLen));
      }
      patch.copyTo(dstImg(Range(y, y + edgeLen), Range(x, x + edgeLen)));
    }
  }
  // TODO: process area that haven't been filled
}

int main(int argc, char **argv) {
  Mat srcImg = cv::imread("./imgs/btile.tif");
  int edgeLen = 20;
  int overlap = 5;
  Size targetSize(250, 250);
  std::cout << "srcImg:" << srcImg.size() << std::endl;

  Mat dstImg;
  synthesis(srcImg, edgeLen, overlap, targetSize, dstImg);

  cv::imwrite("./imgs/output.jpg", dstImg);
  return 0;
}