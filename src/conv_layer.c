#include "conv_layer.h"

int N_ROWS_CONV;
int N_COLS_CONV;

void convolution(tensor* input_t, tensor* conv_t, int n_rows, int n_cols, int batch_size,
	tensor* fil_w, tensor* fil_b, int base, int shuffle_index[]){

	for (int b = 0; b < batch_size; ++b)
	{
		COST_INC_I_ADD(1); //b++
		for (int f = 0; f < NUM_FILS; ++f)
		{
			COST_INC_I_ADD(1); //f++
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				COST_INC_I_ADD(1); //i++
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					COST_INC_I_ADD(2); //j++ & base+b

					(conv_t->data)[offset(conv_t,b,j,i,f)] = convolve(input_t, i, j, b+base, fil_w, fil_b, f, shuffle_index);
				}
			}

		}
	}
}

void bin_convolution(tensor* input_t, tensor* conv_t, int n_rows, int n_cols, int batch_size,
	int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[]){

	for (int b = 0; b < batch_size; ++b)
	{
		COST_INC_I_ADD(1); // b++
		for (int f = 0; f < NUM_FILS; ++f)
		{
			COST_INC_I_ADD(1); // f++
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				COST_INC_I_ADD(1); // i++
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					COST_INC_I_ADD(2); // j++ & b+base
					(conv_t->data)[offset(conv_t,b,j,i,f)] = bin_convolve(input_t, i, j, b+base,
																fil_bin_w, alphas, fil_b, f, shuffle_index);
				}
			}

		}
	}
}

void xnor_convolution(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV],
						tensor* conv_t, int n_rows, int n_cols, int batch_size, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS],
						double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[]){
	for (int b = 0; b < batch_size; ++b)
	{
		COST_INC_I_ADD(1); // b++
		for (int f = 0; f < NUM_FILS; ++f)
		{
			COST_INC_I_ADD(1); // f++
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				COST_INC_I_ADD(1); // i++
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					COST_INC_I_ADD(1); // j++
					(conv_t->data)[offset(conv_t,b,j,i,f)] = xnor_convolve(bin_input_images, betas, i, j, b,
																fil_bin_w, alphas, fil_b, f);
				}
			}

		}
	}
}

void initialize_filters(tensor* fil_w, tensor* fil_b) {
    srand(time(NULL));

    for(int k = 0; k < NUM_FILS; k++){
			COST_INC_I_ADD(1); // k++
        (fil_b->data)[offset(fil_b, k, 0, 0, 0)] = 0.0;
        for(int i = 0; i < FIL_ROWS; ++i){
					COST_INC_I_ADD(1); // i++
            for(int j = 0; j < FIL_COLS; ++j){
							COST_INC_I_ADD(1); // j++
                for(int l = 0; l < FIL_DEPTH; ++l){
									COST_INC_I_ADD(1); // l++
                    int r = rand();
										COST_INC_F_DIV(1);
                    double ran = ((double)rand())/RAND_MAX;
										COST_INC_I_OTHER(1);
                    if (r%2 == 0){
												COST_INC_F_MUL(1);
                        (fil_w->data)[offset(fil_w, k, j, i, l)] = -ran;
                    }else{
                        (fil_w->data)[offset(fil_w, k, j, i, l)] = ran;
                    }
                }
            }
        }
    }
}

void print_filters(tensor* fil_w, tensor* fil_b){
	for (int k = 0; k < NUM_FILS; ++k)
	{
		COST_INC_I_ADD(1); // K++
		printf("k=%d, bias=%f\nweights:\n", k, (fil_b->data)[offset(fil_b, k, 0, 0, 0)]);
		for (int i = 0; i < FIL_ROWS; ++i)
		{
			COST_INC_I_ADD(1); // i++
			for (int j = 0; j < FIL_COLS; ++j)
			{
				COST_INC_I_ADD(1); // j++
				printf("%.3f, ", (fil_w->data)[offset(fil_w, k, j, i, 0)]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void print_bin_filters(int bin_fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS]){
	for (int k = 0; k < NUM_FILS; ++k)
	{
		COST_INC_I_ADD(1);
		printf("k=%d, alpha=%f\nweights:\n", k, alphas[k]);
		for (int i = 0; i < FIL_ROWS; ++i)
		{
			COST_INC_I_ADD(1);
			for (int j = 0; j < FIL_COLS; ++j)
			{
				COST_INC_I_ADD(1);
				printf("%3d, ", bin_fil_w[k][i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

// Tested!!
double convolve(tensor* t, int r, int c, int image_num, tensor* fil_w, tensor* fil_b, int f, int shuffle_index[]){

	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		COST_INC_I_ADD(1);
		for (int j = 0; j < FIL_COLS; ++j)
		{
			COST_INC_I_ADD(1);
        for (int k = 0; k < FIL_DEPTH; ++k){
					COST_INC_I_ADD(1);

					COST_INC_F_ADD(1); COST_INC_F_MUL(1); COST_INC_I_ADD(2);
			    conv_val += (fil_w->data)[offset(fil_w, f, j, i, k)] * (t->data)[offset(t, shuffle_index[image_num], c+j, r+i, k)];
            }
		}
	}

	COST_INC_F_ADD(1);
	conv_val += (fil_b->data)[offset(fil_b, f, 0, 0, 0)];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}

double bin_convolve(tensor* t, int r, int c, int image_num, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS],
	tensor fil_b, int f, int shuffle_index[]){

	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		COST_INC_I_ADD(1);
		for (int j = 0; j < FIL_COLS; ++j)
		{
			COST_INC_I_ADD(1);
			if (fil_bin_w[f][i][j] == 1)
			{
				COST_INC_F_ADD(1); COST_INC_I_ADD(2);
				conv_val += (t->data)[offset(t, shuffle_index[image_num], c+j, r+i, 0)];
			}
			else
			{
				COST_INC_F_ADD(1); COST_INC_I_ADD(2);
				conv_val -= (t->data)[offset(t, shuffle_index[image_num], c+j, r+i, 0)];
			}
		}
	}
	COST_INC_F_MUL(1);
	conv_val *= alphas[f];
	COST_INC_F_ADD(1);
	conv_val += fil_b.data[offset(&fil_b, f, 0, 0, 0)];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}

double xnor_convolve(int t[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV],
						int r, int c, int batch, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS],
						double alphas[NUM_FILS], tensor fil_b, int f){

	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		COST_INC_I_ADD(1);
		for (int j = 0; j < FIL_COLS; ++j)
		{
			COST_INC_I_ADD(1);
			// XNOR operation
			//conv_val += ( t[batch][i+r][j+c] == fil_bin_w[f][i][j] );
			COST_INC_F_ADD(1); COST_INC_I_ADD(2); COST_INC_I_MUL(1);
			conv_val += ( t[batch][i+r][j+c] * fil_bin_w[f][i][j] );
		}
	}

	COST_INC_F_MUL(2);
	conv_val *= alphas[f]*betas[batch][r][c];
	COST_INC_F_ADD(1);
	conv_val += fil_b.data[offset(&fil_b, f, 0, 0, 0)];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}
