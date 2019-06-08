#pragma once

#include "common.h"
#include "growImg.h"

void stichImg(const Mat &imgSrc, const Mat &imgDst, const Mat &dstMask,
              Mat &outputImg);
void testGrowImg();
void testGrowImg2();
void testGrowImg3();
void testGrowImg4();
void testGrowImg5();
void testFindMatches();
void testGetUnfilledNeighbors();