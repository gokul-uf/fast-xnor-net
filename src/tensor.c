#include "tensor.h"

void build_args(tensor * t, int width, int height, int depth, int batch_size){
    t->width        = width;
    t->height       = height;
    t->depth        = depth;
    t->batch_size   = batch_size;
    t->data         = malloc(width * height * depth * batch_size * sizeof(int));
}

void build_batch(tensor * t, int batch_size){
    build_args(t, T_WIDTH, T_HEIGHT, T_DEPTH, batch_size);
}

void build(tensor * t){
    build_args(t, T_WIDTH, T_HEIGHT, T_DEPTH, T_BATCH_SIZE);
}

void destroy(tensor * t){
    free(t->data);
}

int offset( tensor * t, int b, int w, int h, int d ) {
    int width       = t->width;
    int height      = t->height;
    int depth       = t->depth;
    
    return ( b * width * height * depth ) + ( w * height * depth ) + ( h * depth ) + d;
}
