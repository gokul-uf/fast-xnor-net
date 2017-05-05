#include "main.h"
#include "perf.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int N_ROWS_POOL;
int N_COLS_POOL;

int number_of_images;
int number_of_labels;
int n_rows;
int n_cols;

int n_batches;
int* shuffle_index;

int num_val;
int num_train;

// dimension: 60k*28*28
tensor input_images;
int* labels;

// double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS];
// double fil_b[NUM_FILS];

tensor fil_w;
tensor fil_b;

// dimension: BATCH_SIZE*24*24
tensor conv_t;

// dimension: BATCH_SIZE*12*12
tensor pool_t;


tensor fully_con_w, fully_con_b, fully_con_out;

tensor softmax_out;

void shuffle(int shuffle_index[], int number_of_images);

int main(){
    perf_init();

    printf("starting program\n");

    //test_tensor();
    //test_reverse_int();

    set_paths();

    read_mnist_images_labels(TRAIN_IMAGES, TRAIN_LABELS, &number_of_images, &number_of_labels,
    	&n_rows, &n_cols, &input_images, &labels);

    printf("number_of_images=%d\n", number_of_images);
    printf("number_of_labels=%d\n", number_of_labels);
    printf("n_rows in each image=%d\n", n_rows);
    printf("n_cols in each image=%d\n\n", n_cols);

    //test_mnist_load(input_images, labels, number_of_images);

    // Now we have input layer as input_images. Next is convolution layer

    build_args(&fil_w, FIL_COLS, FIL_ROWS, FIL_DEPTH, NUM_FILS);
    build_args(&fil_b, 1, 1, 1, NUM_FILS);
    initialize_filters(&fil_w, &fil_b);
    print_filters(&fil_w, &fil_b);

    N_ROWS_CONV = n_rows - FIL_ROWS + 1;
    N_COLS_CONV = n_cols - FIL_COLS + 1;
    // dimension: BATCH_SIZE*24*24
    build_args(&conv_t, N_COLS_CONV, N_ROWS_CONV, NUM_FILS, BATCH_SIZE);


    N_ROWS_POOL = N_ROWS_CONV/POOL_DIM;
    N_COLS_POOL = N_COLS_CONV/POOL_DIM;

    int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL];
    int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL];

    // dimension: BATCH_SIZE*12*12
    build_args(&pool_t, N_COLS_POOL, N_ROWS_POOL, NUM_FILS, BATCH_SIZE);

    // fully connected layer
    build_args(&fully_con_w, N_COLS_POOL, N_ROWS_POOL, NUM_FILS, N_DIGS);
    build_args(&fully_con_b, 1, 1, N_DIGS, 1);
    build_args(&fully_con_out, 1, 1, N_DIGS, BATCH_SIZE);

    // softmax layer
    build_args(&softmax_out, 1, 1, N_DIGS, BATCH_SIZE);

    initialize_weights_biases(&fully_con_w, &fully_con_b);

    // backprop to max-pool layer
    tensor del_max_pool;
    build_args(&del_max_pool, N_COLS_POOL, N_ROWS_POOL, NUM_FILS, BATCH_SIZE);

    // backprop to conv layer
    tensor del_conv;
    build_args(&del_conv, N_COLS_CONV, N_ROWS_CONV, NUM_FILS, BATCH_SIZE);


    // accuracy on taining set
    int preds[BATCH_SIZE];
    double train_acc = 0.0;
    int correct_preds = 0;

    // accuracy on validation set
    double val_acc = 0.0;

    num_val = NUM_VAL;
    num_train = number_of_images - num_val;

    // First 50k indexes for training, next 10k indexes for validation
    shuffle_index = malloc(number_of_images*sizeof(int));

    n_batches = num_train/BATCH_SIZE;


    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch)
    {
        // Shuffle all 60 only once, they keep last 10 for validtion
        if (epoch == 0)
        {
            shuffle(shuffle_index, number_of_images);
        }
        else
        {
            shuffle(shuffle_index, num_train);
        }

        correct_preds = 0;
        for (int i = 0; i < n_batches; ++i)
        {
    	    convolution(&input_images, &conv_t, n_rows, n_cols, BATCH_SIZE, &fil_w, &fil_b, i*BATCH_SIZE, shuffle_index);
    	    max_pooling(&conv_t, &pool_t, pool_index_i, pool_index_j, BATCH_SIZE, 'T');
    	    feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b, BATCH_SIZE);
    	    softmax(&fully_con_out, &softmax_out, preds, BATCH_SIZE);

    	    bp_softmax_to_maxpool(&del_max_pool, softmax_out, labels, i*BATCH_SIZE, fully_con_w, shuffle_index);
    	    bp_maxpool_to_conv(&del_conv, del_max_pool, conv_t, pool_index_i, pool_index_j);

    	    // update weights and biases
    	    update_sotmax_weights(&fully_con_w, softmax_out, pool_t, labels, i*BATCH_SIZE, shuffle_index);
    	    update_sotmax_biases(&fully_con_b, softmax_out, labels, i*BATCH_SIZE, shuffle_index);
    	    update_conv_weights(&fil_w, del_conv, conv_t, input_images, i*BATCH_SIZE, shuffle_index);
    	    update_conv_biases(&fil_b, del_conv, conv_t);

            correct_preds += calc_correct_preds(preds, labels, i, shuffle_index);

            if( (i+1)%500 == 0 ){
                train_acc = (correct_preds*100.0) / ((i+1)*BATCH_SIZE);

                val_acc = validate();

                printf("\nEpoch=%3d, Batch=%3d, train_acc=%3.2f% val_acc=%3.2f% \n", epoch+1, i+1, train_acc, val_acc);
                /*printf("\nPred\n");
                print_tensor_1d(&softmax_out, 10, 0);
                printf("Label: %d\n", labels[ shuffle_index[i*BATCH_SIZE] ]);*/

                //print_filters(fil_w, fil_b);
                //print_tensor_1d(&softmax_out, 10, 0);
                //print_tensor(&fully_con_w, 0, 12);
            }

    	    reset_to_zero(&del_max_pool);
    	    reset_to_zero(&del_conv);
    	    reset_to_zero(&conv_t);
    	    reset_to_zero(&pool_t);
    	    reset_to_zero(&fully_con_out);
    	    reset_to_zero(&softmax_out);
        }
}


    // testing convolution with last image in last batch
    //print_tensor(&input_images, shuffle_index[0], 28);
    //print_tensor(&conv_t, 0, 24);
    //print_tensor(&pool_t, 0, 12);

    //print_pool_mat(pool_index_i, pool_index_j, 99);

    //print_tensor_1d(&fully_con_out, 10, 99);
    //print_tensor_1d(&softmax_out, 10, 99);

    destroy(&input_images);
    destroy(&conv_t);
    destroy(&pool_t);
    destroy(&fully_con_w);
    destroy(&fully_con_b);
    destroy(&fully_con_out);
    destroy(&softmax_out);
    destroy(&del_conv);
    destroy(&del_max_pool);

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

int calc_correct_preds(int preds[BATCH_SIZE], int* labels, int num_batch, int shuffle_index[]){
    int base = BATCH_SIZE*num_batch;
    int ret = 0;
    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        if (preds[i] == labels[ shuffle_index[base+i] ])
        {
            ret++;
        }
    }

    return ret;
}


void shuffle(int shuffle_index[], int number_of_images){
    srand( time(NULL) );

    for (int i = 0; i < number_of_images; ++i)
    {
        shuffle_index[i] = i;
    }

    for (int i = number_of_images-1; i >= 0; --i){
    //generate a random number [0, n-1]
    int j = rand() % (i+1);

    //swap the last element with element at random index
    int temp = shuffle_index[i];
    shuffle_index[i] = shuffle_index[j];
    shuffle_index[j] = temp;
    }
}

double validate(){

    int pred[1];
    int correct_preds = 0;
    for (int i = num_train; i < num_train + num_val; ++i)
        {
            convolution(&input_images, &conv_t, n_rows, n_cols, 1, &fil_w, &fil_b, i, shuffle_index);
            max_pooling(&conv_t, &pool_t, NULL, NULL, 1, 'V');
            feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b, 1);
            softmax(&fully_con_out, &softmax_out, pred, 1);

            correct_preds += (labels[shuffle_index[i]] == pred[0]);
        }

    return (correct_preds*100.0) / num_val;
}


