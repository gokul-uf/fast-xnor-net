#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"
#include "common.h"

double convolve(tensor* t, int r, int c, int image_num, tensor* fil_w, tensor* fil_b, int f, int shuffle_index[]);
double bin_convolve(tensor* t, int r, int c, int image_num, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS],
	tensor fil_b, int f, int shuffle_index[]);
double xnor_convolve(int t[NUM_IMAGES][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV],
						int r, int c, int b, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
						double alphas[NUM_FILS], tensor fil_b, int f);

#endif


