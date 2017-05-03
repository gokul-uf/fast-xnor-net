#include "main.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int N_ROWS_POOL;
int N_COLS_POOL;

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

    // dimension: BATCH_SIZE*12*12
    tensor pool_t;

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


    N_ROWS_CONV = n_rows - FIL_ROWS + 1;
    N_COLS_CONV = n_cols - FIL_COLS + 1;
    // dimension: BATCH_SIZE*24*24
    build_args(&conv_t, N_COLS_CONV, N_ROWS_CONV, NUM_FILS, BATCH_SIZE);


    N_ROWS_POOL = N_ROWS_CONV/POOL_DIM;
    N_COLS_POOL = N_COLS_CONV/POOL_DIM;

    // dimension: BATCH_SIZE*12*12
    build_args(&pool_t, N_COLS_POOL, N_ROWS_POOL, NUM_FILS, BATCH_SIZE);
    int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL];
    int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL];


    // fully connected layer
    tensor fully_con_w, fully_con_b, fully_con_out;

    build_args(&fully_con_w, N_COLS_POOL, N_ROWS_POOL, NUM_FILS, N_DIGS);
    build_args(&fully_con_b, 1, 1, N_DIGS, 1);
    build_args(&fully_con_out, 1, 1, N_DIGS, BATCH_SIZE);

    // softmax layer
    tensor softmax_out;
    build_args(&softmax_out, 1, 1, N_DIGS, BATCH_SIZE);

    initialize_weights_biases(&fully_con_w, &fully_con_b);

    n_batches = number_of_images/BATCH_SIZE;
    for (int i = 0; i < n_batches; ++i)
    {
	    convolution(&input_images, &conv_t, n_rows, n_cols, fil_w, fil_b, i*BATCH_SIZE);
	    max_pooling(&conv_t, &pool_t, pool_index_i, pool_index_j);
	    feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b);
	    softmax(&fully_con_out, &softmax_out);
	    update_sotmax_weights(&fully_con_w, &fully_con_b, softmax_out, pool_t, labels, i*BATCH_SIZE);
    }


    // testing convolution with last image in last batch
    print_tensor(&input_images, 59999, 28);
    print_tensor(&conv_t, 99, 24);
    print_tensor(&pool_t, 99, 12);

    //print_pool_mat(pool_index_i, pool_index_j, 99);

    print_tensor_1d(&fully_con_out, 10, 99);
    print_tensor_1d(&softmax_out, 10, 99);

    return 0;
}

void print_pool_mat(int mat1[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int mat2[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int num){
	printf("Max Pooling: %d\n", num);
	for (int i = 0; i < N_ROWS_POOL; ++i)
	{
		for (int j = 0; j < N_COLS_POOL; ++j)
		{
			printf("%2d-%2d, ", mat1[num][0][i][j], mat2[num][0][i][j]);				
		}
		printf("\n");
	}
	printf("\n");
}




