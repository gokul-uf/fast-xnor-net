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

void test_tensor()
{
    printf("\nIn test_tensor:\n\n");
    tensor t;
    build(&t);
    t.data[offset(&t,0,0,0,0)] = 1;
    t.data[offset(&t,0,0,1,0)] = 4;
    t.data[offset(&t,0,1,0,0)] = 3;
    t.data[offset(&t,0,1,1,0)] = 4;
    
    printf("%d %d %d %d ---> %d x %d x %d x %d \n", t.data[0], t.data[1], t.data[2], t.data[3],
           t.width, t.height, t.depth, t.batch_size);
    destroy(&t);

    printf("\ntest_tensor done!\n\n");
}