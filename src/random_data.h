#ifndef RANDOM_DATA_H
#define RANDOM_DATA_H

#include "tensor.h"
#include "common.h"
#include <time.h>
#include <stdlib.h>

void random_data(int image_cols, int image_rows, tensor* input_tensor, int** labels);

#endif
