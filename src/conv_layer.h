#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"
#include "common.h"

double convolve(tensor* t, int r, int c, int image_num, double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double fil_b[NUM_FILS], int f, int shuffle_index[]);

#endif


