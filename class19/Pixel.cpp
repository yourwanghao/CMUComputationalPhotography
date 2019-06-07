#include "Pixel.h"

bool cmp(const Pixel &a, const Pixel &b) {
  if (a.count < b.count) {
    return true;
  } else {
    return false;
  }
}

void Pixel::show() {
  printf("(%d, %d): count=%d, value=%d, error=%f\n", y, x, count, value, error);
}

Pixel::Pixel(int y, int x, int count) : y(y), x(x), count(count) {}
