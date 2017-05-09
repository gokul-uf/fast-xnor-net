#include "main.h"
#include "perf.h"

int NUM_IMAGES;
int IMAGE_ROWS;
int IMAGE_COLS;

int N_ROWS_CONV;
int N_COLS_CONV;
int N_ROWS_POOL;
int N_COLS_POOL;

int n_batches;
int* shuffle_index;

int num_val;
int num_train;

// dimension: 60k*28*28
tensor input_images;
int* labels;

tensor fil_w;
tensor fil_b;

// dimension: BATCH_SIZE*24*24
tensor conv_t;

// dimension: BATCH_SIZE*12*12
tensor pool_t;

int**** pool_index_i;
int**** pool_index_j;

tensor fully_con_w, fully_con_b, fully_con_out;
tensor softmax_out;
tensor del_max_pool;
tensor del_conv;

// binarized input batch of images
int*** bin_input_images;
double*** betas;

// binarized weights in conv layer
int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS];
double alphas[NUM_FILS];

// accuracy on taining set
int preds[BATCH_SIZE];
double train_acc = 0.0;
int correct_preds = 0;

// accuracy on validation set
double val_acc = 0.0;

// perf counters
int64_t conv_cycles = 0;
int64_t binarize_cycles = 0;
int64_t xnor_cycles = 0;
int64_t pool_cycles = 0;
int64_t fully_cycles = 0;
int64_t soft_cycles = 0;
int64_t bp_soft_to_pool_cycles = 0;
int64_t bp_pool_to_conv_cycles = 0;
int64_t forward_cycles = 0;
int64_t backward_cycles = 0;
int64_t total_cycles = 0;

int main(){
    perf_init();

    printf("starting program\n");

    //test_tensor();
    //test_reverse_int();

    set_paths();

    read_mnist_images_labels(TRAIN_IMAGES, TRAIN_LABELS, &input_images, &labels);

    printf("number_of_images=%d\n", NUM_IMAGES);
    printf("number_of_labels=%d\n", NUM_IMAGES);
    printf("n_rows in each image=%d\n", IMAGE_ROWS);
    printf("n_cols in each image=%d\n\n", IMAGE_COLS);

    build_args(&fil_w, FIL_COLS, FIL_ROWS, FIL_DEPTH, NUM_FILS);
    build_args(&fil_b, 1, 1, 1, NUM_FILS);
    initialize_filters(&fil_w, &fil_b);

    N_ROWS_CONV = IMAGE_ROWS - FIL_ROWS + 1;
    N_COLS_CONV = IMAGE_COLS - FIL_COLS + 1;
    // dimension: BATCH_SIZE*24*24
    build_args(&conv_t, N_COLS_CONV, N_ROWS_CONV, NUM_FILS, BATCH_SIZE);


    N_ROWS_POOL = N_ROWS_CONV/POOL_DIM;
    N_COLS_POOL = N_COLS_CONV/POOL_DIM;

    int index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL];
    int index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL];

    pool_index_i = index_i;
    pool_index_j = index_j;

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
    build_args(&del_max_pool, N_COLS_POOL, N_ROWS_POOL, NUM_FILS, BATCH_SIZE);

    // backprop to conv layer
    build_args(&del_conv, N_COLS_CONV, N_ROWS_CONV, NUM_FILS, BATCH_SIZE);

    num_val = NUM_VAL;
    num_train = NUM_IMAGES - num_val;

    // First 50k indexes for training, next 10k indexes for validation
    shuffle_index = malloc(NUM_IMAGES*sizeof(int));

    n_batches = num_train/BATCH_SIZE;

    if (BINARY_NET == 0)
    {
        normal_net();
    }
    else if (BINARY_NET == 1)
    {
        binary_net();
    }
    else
    {
        int bin_array[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS];
        bin_input_images = bin_array;

        double betas_array[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV];
        betas = betas_array;

        xnor_net();
    }

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

void normal_net()
{

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch)
    {
        // Shuffle all 60k only once, they keep last 10k for validtion
        conv_cycles = 0;
        pool_cycles = 0;
        fully_cycles = 0;
        soft_cycles = 0;
        bp_soft_to_pool_cycles = 0;
        bp_pool_to_conv_cycles = 0;
        forward_cycles = 0;
        backward_cycles = 0;
        total_cycles = 0;
        if (epoch == 0)
        {
            shuffle(shuffle_index, NUM_IMAGES);
        }
        else
        {
            shuffle(shuffle_index, num_train);
        }

        correct_preds = 0;
        for (int i = 0; i < n_batches; ++i)
        {
            cycles_count_start();
    	    convolution(&input_images, &conv_t, BATCH_SIZE, &fil_w, &fil_b, i*BATCH_SIZE, shuffle_index);
            conv_cycles += cycles_count_stop();

            cycles_count_start();
    	    max_pooling(&conv_t, &pool_t, pool_index_i, pool_index_j, BATCH_SIZE, 'T');
            pool_cycles += cycles_count_stop();

            cycles_count_start();
    	    feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b, BATCH_SIZE);
            fully_cycles += cycles_count_stop();

            cycles_count_start();
    	    softmax(&fully_con_out, &softmax_out, preds, BATCH_SIZE);
            soft_cycles += cycles_count_stop();

            cycles_count_start();
    	    bp_softmax_to_maxpool(&del_max_pool, &softmax_out, labels, i*BATCH_SIZE, &fully_con_w, shuffle_index);
    	    update_sotmax_weights(&fully_con_w, &softmax_out, &pool_t, labels, i*BATCH_SIZE, shuffle_index);
    	    update_sotmax_biases(&fully_con_b, &softmax_out, labels, i*BATCH_SIZE, shuffle_index);
            bp_soft_to_pool_cycles += cycles_count_stop();

            cycles_count_start();
    	    bp_maxpool_to_conv(&del_conv, &del_max_pool, &conv_t, pool_index_i, pool_index_j);
    	    update_conv_weights(&fil_w, &del_conv, &conv_t, &input_images, i*BATCH_SIZE, shuffle_index);
    	    update_conv_biases(&fil_b, &del_conv, &conv_t);
            bp_pool_to_conv_cycles += cycles_count_stop();

            reset_to_zero(&del_max_pool);
            reset_to_zero(&del_conv);
            reset_to_zero(&conv_t);
            reset_to_zero(&pool_t);
            reset_to_zero(&fully_con_out);
            reset_to_zero(&softmax_out);
        }
        forward_cycles = conv_cycles + pool_cycles + fully_cycles + soft_cycles;
        backward_cycles = bp_soft_to_pool_cycles + bp_pool_to_conv_cycles;
        total_cycles = forward_cycles + backward_cycles;

        printf("epoch %d:\n", epoch+1);
        printf("conv_cycles: %lld\n", conv_cycles);
        printf("pool_cycles: %lld\n", pool_cycles);
        printf("fully_cycles: %lld\n", fully_cycles);
        printf("soft_cycles: %lld\n", soft_cycles);
        printf("bp_soft_to_pool_cycles: %lld\n", bp_soft_to_pool_cycles);
        printf("bp_pool_to_conv_cycles: %lld\n", bp_pool_to_conv_cycles);
        printf("forward_cycles: %lld\n", forward_cycles);
        printf("backward_cycles: %lld\n", backward_cycles);
        printf("total_cycles: %lld\n", total_cycles);
        printf("\n");
    }


}

void binary_net()
{

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch)
    {
        binarize_cycles = 0;
        conv_cycles = 0;
        pool_cycles = 0;
        fully_cycles = 0;
        soft_cycles = 0;
        bp_soft_to_pool_cycles = 0;
        bp_pool_to_conv_cycles = 0;
        forward_cycles = 0;
        backward_cycles = 0;
        total_cycles = 0;
        // Shuffle all 60k only once, they keep last 10k for validtion
        if (epoch == 0)
        {
            shuffle(shuffle_index, NUM_IMAGES);
        }
        else
        {
            shuffle(shuffle_index, num_train);
        }

        correct_preds = 0;
        for (int i = 0; i < n_batches; ++i)
        {
            cycles_count_start();
            binarize_filters(&fil_w, fil_bin_w, alphas);
            binarize_cycles += cycles_count_stop();

            cycles_count_start();
            bin_convolution(&input_images, &conv_t, BATCH_SIZE, fil_bin_w, alphas,
                fil_b, i*BATCH_SIZE, shuffle_index);
            conv_cycles += cycles_count_stop();

            cycles_count_start();
    	    max_pooling(&conv_t, &pool_t, pool_index_i, pool_index_j, BATCH_SIZE, 'T');
            pool_cycles += cycles_count_stop();

            cycles_count_start();
    	    feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b, BATCH_SIZE);
            fully_cycles += cycles_count_stop();

            cycles_count_start();
    	    softmax(&fully_con_out, &softmax_out, preds, BATCH_SIZE);
            soft_cycles += cycles_count_stop();

            cycles_count_start();
    	    bp_softmax_to_maxpool(&del_max_pool, &softmax_out, labels, i*BATCH_SIZE, &fully_con_w, shuffle_index);
    	    update_sotmax_weights(&fully_con_w, &softmax_out, &pool_t, labels, i*BATCH_SIZE, shuffle_index);
    	    update_sotmax_biases(&fully_con_b, &softmax_out, labels, i*BATCH_SIZE, shuffle_index);
            bp_soft_to_pool_cycles += cycles_count_stop();

            cycles_count_start();
    	    bp_maxpool_to_conv(&del_conv, &del_max_pool, &conv_t, pool_index_i, pool_index_j);
    	    update_conv_weights(&fil_w, &del_conv, &conv_t, &input_images, i*BATCH_SIZE, shuffle_index);
    	    update_conv_biases(&fil_b, &del_conv, &conv_t);
            bp_pool_to_conv_cycles += cycles_count_stop();

            reset_to_zero(&del_max_pool);
            reset_to_zero(&del_conv);
            reset_to_zero(&conv_t);
            reset_to_zero(&pool_t);
            reset_to_zero(&fully_con_out);
            reset_to_zero(&softmax_out);
        }
        forward_cycles = binarize_cycles + conv_cycles + pool_cycles + fully_cycles + soft_cycles;
        backward_cycles = bp_soft_to_pool_cycles + bp_pool_to_conv_cycles;
        total_cycles = forward_cycles + backward_cycles;

        printf("epoch %d:\n", epoch+1);
        printf("binarize_cycles: %lld\n", binarize_cycles);
        printf("conv_cycles: %lld\n", conv_cycles);
        printf("pool_cycles: %lld\n", pool_cycles);
        printf("fully_cycles: %lld\n", fully_cycles);
        printf("soft_cycles: %lld\n", soft_cycles);
        printf("bp_soft_to_pool_cycles: %lld\n", bp_soft_to_pool_cycles);
        printf("bp_pool_to_conv_cycles: %lld\n", bp_pool_to_conv_cycles);
        printf("forward_cycles: %lld\n", forward_cycles);
        printf("backward_cycles: %lld\n", backward_cycles);
        printf("total_cycles: %lld\n", total_cycles);
        printf("\n");
    }
}

void xnor_net()
{

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch)
    {
        xnor_cycles = 0;
        conv_cycles = 0;
        pool_cycles = 0;
        fully_cycles = 0;
        soft_cycles = 0;
        bp_soft_to_pool_cycles = 0;
        bp_pool_to_conv_cycles = 0;
        forward_cycles = 0;
        backward_cycles = 0;
        total_cycles = 0;
        // Shuffle all 60k only once, they keep last 10k for validtion
        if (epoch == 0)
        {
            shuffle(shuffle_index, NUM_IMAGES);
        }
        else
        {
            shuffle(shuffle_index, num_train);
        }

        correct_preds = 0;
        for (int i = 0; i < n_batches; ++i)
        {

            cycles_count_start();
            binarize_filters(&fil_w, fil_bin_w, alphas);
            bin_activation(&input_images, bin_input_images, shuffle_index, betas, BATCH_SIZE, i*BATCH_SIZE);
            xnor_cycles += cycles_count_stop();

            cycles_count_start();
            xnor_convolution(bin_input_images, betas, &conv_t, BATCH_SIZE, fil_bin_w, alphas,
                fil_b, i*BATCH_SIZE, shuffle_index);
            conv_cycles += cycles_count_stop();

            cycles_count_start();
    	    max_pooling(&conv_t, &pool_t, pool_index_i, pool_index_j, BATCH_SIZE, 'T');
            pool_cycles += cycles_count_stop();

            cycles_count_start();
    	    feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b, BATCH_SIZE);
            fully_cycles += cycles_count_stop();

            cycles_count_start();
    	    softmax(&fully_con_out, &softmax_out, preds, BATCH_SIZE);
            soft_cycles += cycles_count_stop();

            cycles_count_start();
    	    bp_softmax_to_maxpool(&del_max_pool, &softmax_out, labels, i*BATCH_SIZE, &fully_con_w, shuffle_index);
    	    update_sotmax_weights(&fully_con_w, &softmax_out, &pool_t, labels, i*BATCH_SIZE, shuffle_index);
    	    update_sotmax_biases(&fully_con_b, &softmax_out, labels, i*BATCH_SIZE, shuffle_index);
            bp_soft_to_pool_cycles += cycles_count_stop();

            cycles_count_start();
    	    bp_maxpool_to_conv(&del_conv, &del_max_pool, &conv_t, pool_index_i, pool_index_j);
    	    update_conv_weights(&fil_w, &del_conv, &conv_t, &input_images, i*BATCH_SIZE, shuffle_index);
    	    update_conv_biases(&fil_b, &del_conv, &conv_t);
            bp_pool_to_conv_cycles += cycles_count_stop();

            reset_to_zero(&del_max_pool);
            reset_to_zero(&del_conv);
            reset_to_zero(&conv_t);
            reset_to_zero(&pool_t);
            reset_to_zero(&fully_con_out);
            reset_to_zero(&softmax_out);
        }
        forward_cycles = xnor_cycles + conv_cycles + pool_cycles + fully_cycles + soft_cycles;
        backward_cycles = bp_soft_to_pool_cycles + bp_pool_to_conv_cycles;
        total_cycles = forward_cycles + backward_cycles;

        printf("epoch %d:\n", epoch+1);
        printf("xnor_cycles: %lld\n", xnor_cycles);
        printf("conv_cycles: %lld\n", conv_cycles);
        printf("pool_cycles: %lld\n", pool_cycles);
        printf("fully_cycles: %lld\n", fully_cycles);
        printf("soft_cycles: %lld\n", soft_cycles);
        printf("bp_soft_to_pool_cycles: %lld\n", bp_soft_to_pool_cycles);
        printf("bp_pool_to_conv_cycles: %lld\n", bp_pool_to_conv_cycles);
        printf("forward_cycles: %lld\n", forward_cycles);
        printf("backward_cycles: %lld\n", backward_cycles);
        printf("total_cycles: %lld\n", total_cycles);
        printf("\n");
    }
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
            convolution(&input_images, &conv_t, 1, &fil_w, &fil_b, i, shuffle_index);
            max_pooling(&conv_t, &pool_t, NULL, NULL, 1, 'V');
            feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b, 1);
            softmax(&fully_con_out, &softmax_out, pred, 1);

            correct_preds += (labels[shuffle_index[i]] == pred[0]);
        }

    return (correct_preds*100.0) / num_val;
}

double bin_validate(){

    int pred[1];
    int correct_preds = 0;
    for (int i = num_train; i < num_train + num_val; ++i)
        {
            bin_convolution(&input_images, &conv_t, 1, fil_bin_w, alphas, fil_b, i, shuffle_index);
            max_pooling(&conv_t, &pool_t, NULL, NULL, 1, 'V');
            feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b, 1);
            softmax(&fully_con_out, &softmax_out, pred, 1);

            correct_preds += (labels[shuffle_index[i]] == pred[0]);
        }

    return (correct_preds*100.0) / num_val;
}

double xnor_validate(){

    int pred[1];
    int correct_preds = 0;
    for (int i = num_train; i < num_train + num_val; ++i)
        {
            // calculate betas
            bin_activation(&input_images, bin_input_images, shuffle_index, betas, 1, i);

            xnor_convolution(bin_input_images, betas, &conv_t, 1, fil_bin_w, alphas, fil_b, i, shuffle_index);

            max_pooling(&conv_t, &pool_t, NULL, NULL, 1, 'V');
            feed_forward(&pool_t, &fully_con_out, &fully_con_w, &fully_con_b, 1);
            softmax(&fully_con_out, &softmax_out, pred, 1);

            correct_preds += (labels[shuffle_index[i]] == pred[0]);
        }

    return (correct_preds*100.0) / num_val;
}


