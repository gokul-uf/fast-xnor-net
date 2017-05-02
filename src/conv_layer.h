#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#define NUM_FILS 3
#define FIL_ROWS 5
#define FIL_COLS 5

#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

double convolve(tensor* t, int r, int c, int image_num, double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double fil_b[NUM_FILS], int f);

#endif


