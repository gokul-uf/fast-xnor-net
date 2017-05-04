#include "conv_layer.h"

int N_ROWS_CONV;
int N_COLS_CONV;

void convolution(tensor* input_t, tensor* conv_t, int n_rows, int n_cols, int batch_size,
	double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double fil_b[NUM_FILS], int base, int shuffle_index[]){

	for (int b = 0; b < batch_size; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					(conv_t->data)[offset(conv_t,b,j,i,f)] = convolve(input_t, i, j, b+base, fil_w, fil_b, f, shuffle_index);
				}
			}
			
		}
	}
}

void initialize_filters(double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double fil_b[NUM_FILS]){

	srand( time(NULL) );

	for (int k = 0; k < NUM_FILS; ++k)
	{
		fil_b[k] = 0.0;

		for (int i = 0; i < FIL_ROWS; ++i)
		{
			for (int j = 0; j < FIL_COLS; ++j)
			{
				int r = rand();

				if (r%2 == 0)
				{
					double ran = ((double)rand())/RAND_MAX;
					fil_w[k][i][j] = -ran;					
				}
				else
				{
					double ran = ((double)rand())/RAND_MAX;
					fil_w[k][i][j] = ran;
				}				
			}
		}
	}
}

void print_filters(double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double fil_b[NUM_FILS]){
	for (int k = 0; k < NUM_FILS; ++k)
	{
		printf("k=%d, bias=%f\nweights:\n", k, fil_b[k]);
		for (int i = 0; i < FIL_ROWS; ++i)
		{
			for (int j = 0; j < FIL_COLS; ++j)
			{
				printf("%f, ", fil_w[k][i][j]);				
			}
			printf("\n");
		}
		printf("\n");
	}
}

// Tested!!
double convolve(tensor* t, int r, int c, int image_num, double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
	double fil_b[NUM_FILS], int f, int shuffle_index[]){

	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		for (int j = 0; j < FIL_COLS; ++j)
		{
			conv_val += fil_w[f][i][j] * (t->data)[offset(t, shuffle_index[image_num], c+j, r+i, 0)];
		}
	}

	conv_val += fil_b[f];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}