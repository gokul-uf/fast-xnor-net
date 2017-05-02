#include "conv_layer.h"

int N_ROWS_CONV;
int N_COLS_CONV;

void convolution(tensor* input_t, tensor* conv_t, int n_rows, int n_cols,
	double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double fil_b[NUM_FILS], int base){

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					(conv_t->data)[offset(conv_t,b,j,i,f)] = convolve(input_t, i, j, b+base, fil_w, fil_b, f);
				}
			}
			
		}
	}
}

void initialize_filters(double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double fil_b[NUM_FILS]){
	for (int k = 0; k < NUM_FILS; ++k)
	{
		fil_b[k] = 0.0;

		for (int i = 0; i < FIL_ROWS; ++i)
		{
			for (int j = 0; j < FIL_COLS; ++j)
			{
				fil_w[k][i][j] = 1.0;				
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
double convolve(tensor* t, int r, int c, int image_num, double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double fil_b[NUM_FILS], int f){
	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		for (int j = 0; j < FIL_COLS; ++j)
		{
			conv_val += fil_w[f][i][j] * (t->data)[offset(t,image_num,c+j,r+i,0)];
			/*if (r==7 && c==3 && f==0)
			{
				printf("\n%3.0f, ", conv_val);
			}*/
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