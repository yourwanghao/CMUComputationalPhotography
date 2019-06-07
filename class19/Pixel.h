#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>

class Pixel {
 public:
  Pixel(int y, int x, int count);
  void show();

 public:
  int y;
  int x;
  int count;

  uchar value;
  float error;
};

bool cmp(const Pixel &a, const Pixel &b);


