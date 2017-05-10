#ifndef TENSOR_H
#define TENSOR_H

#define T_WIDTH 2
#define T_HEIGHT 2
#define T_DEPTH 1 // grey-scale images, change to 3 for RGB
#define T_BATCH_SIZE 1

/*#define offset(t, b, w, h, d ) \
    ( ( b * ((t)->width) * ((t)->height) * ((t)->depth) ) + ( w * ((t)->height) * ((t)->depth) ) + ( h * ((t)->depth) ) + d ) \*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int width;
    int height;
    int depth;
    int batch_size;
    double * data;
} tensor;

void build_args(tensor * t, int width, int height, int depth, int batch_size);
void build_batch(tensor * t, int batch_size);
void build(tensor * t);
void destroy(tensor * t);
int offset( tensor * t, int b, int w, int h, int d );
void test_tensor();

#endif
