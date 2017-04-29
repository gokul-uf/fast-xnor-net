#include <stdio.h>
#include "main.h"
#include "xnornet.h"
#include "tensor.h"
#include "mnist_wrapper.h"

char* TEST_IMAGES;
char* TEST_LABELS;
char* TEST_IMAGES;
char* TEST_LABELS;


int main(){
    printf("starting program\n");

    //test_tensor();
    //test_reverse_int();

    int number_of_images;
    int number_of_labels;
    int n_rows;
    int n_cols;
    tensor input_images;
    int* labels;

    set_paths();

    read_mnist_images_labels(TRAIN_IMAGES, TRAIN_LABELS, &number_of_images, &number_of_labels, 
    	&n_rows, &n_cols, &input_images, &labels);

    printf("number_of_images=%d\n", number_of_images);
    printf("number_of_labels=%d\n", number_of_labels);
    printf("n_rows in each image=%d\n", n_rows);
    printf("n_cols in each image=%d\n", n_cols);

    test_mnist_load(input_images, labels, number_of_images);

    return 0;
}




