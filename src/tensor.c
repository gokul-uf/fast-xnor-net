#include "tensor.h"

void build_args(tensor * t, int width, int height, int depth, int batch_size){
    t->width        = width;
    t->height       = height;
    t->depth        = depth;
    t->batch_size   = batch_size;
    t->data         = calloc(width * height * depth * batch_size, sizeof(double));
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

void reset_to_zero(tensor* t){
    memset(t->data, 0, t->width * t->height * t->depth * t->batch_size * sizeof(double));
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
    
    printf("%f %f %f %f ---> %d x %d x %d x %d \n", t.data[0], t.data[1], t.data[2], t.data[3],
           t.width, t.height, t.depth, t.batch_size);
    destroy(&t);

    printf("\ntest_tensor done!\n\n");
}

void test_offset()
{
    printf("\nIn test_tensor:\n\n");
    tensor* t;
    build(t);
    int i = offset(t,0,0,0,0);
    int j = offset(t,0,0,1,0);
    int k = offset(t,0,1,0,0);
    int l = offset(t,0,1,1,0);
    
    printf("%d %d %d %d\n", i, j, k, l);
    destroy(t);

    printf("\ntest_tensor done!\n\n");
}

void print_tensor(tensor* t, int image_num, int len){
    printf("Tensor: %d\n", image_num);
    for (int i = 0; i < len; ++i)
    {
        for (int j = 0; j < len; ++j)
        {
            printf("%4.0f,", (t->data)[offset(t,image_num,j,i,0)]);
        }

        printf("\n\n");
    }
}


void print_tensor_1d(tensor* t, int lim, int b){
    for (int d = 0; d < lim; ++d)
    {
        printf("%f, ", (t->data)[offset(t, b, 0, 0, d)]);
    }

    printf("\n");
}