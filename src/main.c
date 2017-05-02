#include "main.h"

int main(){
    printf("starting program\n");

    //test_tensor();
    //test_reverse_int();

    int number_of_images;
    int number_of_labels;
    int n_rows;
    int n_cols;

    int n_batches;

    // dimension: 60k*28*28
    tensor input_images;
    int* labels;

    double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS];
    double fil_b[NUM_FILS];

    // dimension: BATCH_SIZE*24*24
    tensor conv_t;

    set_paths();

    read_mnist_images_labels(TRAIN_IMAGES, TRAIN_LABELS, &number_of_images, &number_of_labels, 
    	&n_rows, &n_cols, &input_images, &labels);

    printf("number_of_images=%d\n", number_of_images);
    printf("number_of_labels=%d\n", number_of_labels);
    printf("n_rows in each image=%d\n", n_rows);
    printf("n_cols in each image=%d\n\n", n_cols);

    //test_mnist_load(input_images, labels, number_of_images);

    // Now we have input layer as input_images. Next is convolution layer

    initialize_filters(fil_w, fil_b);
    print_filters(fil_w, fil_b);

    // dimension: BATCH_SIZE*24*24
    build_args(&conv_t, n_cols - FIL_COLS + 1, n_rows - FIL_ROWS + 1, NUM_FILS, BATCH_SIZE);


    n_batches = number_of_images/BATCH_SIZE;
    for (int i = 0; i < n_batches; ++i)
    {
	    convolution(&input_images, &conv_t, n_rows, n_cols, fil_w, fil_b, BATCH_SIZE, i*BATCH_SIZE);
    }


    // testing convolution with last image in last batch
    print_tensor(&input_images, 59990, 28);
    print_tensor(&conv_t, 90, 24);

    return 0;
}




