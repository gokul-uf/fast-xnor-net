#include "conv_layer.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int TOTAL_FLOPS;

void convolution(tensor* input_t, tensor* conv_t, int batch_size,
	tensor* fil_w, tensor* fil_b, int base, int shuffle_index[])
{
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

void bin_convolution(tensor* input_t, tensor* conv_t, int batch_size,
	int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[])
{
	for (int b = 0; b < batch_size; b=b+2)
	{
		// load 2 current image
		int cur_image1 = shuffle_index[b+  base];
		int cur_image2 = shuffle_index[b+1+base];

		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					bin_convolve(input_t, conv_t, b, i, j, cur_image1, cur_image2,
																fil_bin_w, alphas, fil_b, f);
				}
			}
			
		}
	}
}

void bin_convolve(tensor* t, tensor* conv_t, int b, int r, int c, int cur_image1, int cur_image2, 
					int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS], tensor fil_b, int f)
{
	double conv_val1, conv_val2, conv_val3, conv_val4, conv_val5, conv_val6, conv_val7, conv_val8;
	double conv_img1, conv_img2, conv_img1_rem = 0.0, conv_img2_rem = 0.0, conv_img1_last = 0.0, conv_img2_last = 0.0;
	conv_val1 = 0.0;
	conv_val2 = 0.0;
	conv_val3 = 0.0;
	conv_val4 = 0.0;
	conv_val5 = 0.0;
	conv_val6 = 0.0;
	conv_val7 = 0.0;
	conv_val8 = 0.0;

	double bias = fil_b.data[offset(&fil_b, f, 0, 0, 0)];

	double mat1, mat2, mat3, mat4;
	double mat5, mat6, mat7, mat8;

	int fil1, fil2, fil3, fil4;

	int i, j;
	for (i = 0; i < FIL_ROWS; i=i+2)
	{
		for (j = 0; j+1 < FIL_COLS; j=j+2)
		{
			INCREMENT_FLOPS(8)

			mat1 = (t->data)[offset(t, cur_image1, c+j  ,   r+i, 0)];
			mat2 = (t->data)[offset(t, cur_image1, c+j+1,   r+i, 0)];
			mat3 = (t->data)[offset(t, cur_image1, c+j,   r+i+1, 0)];
			mat4 = (t->data)[offset(t, cur_image1, c+j+1, r+i+1, 0)];

			mat5 = (t->data)[offset(t, cur_image2, c+j  ,   r+i, 0)];
			mat6 = (t->data)[offset(t, cur_image2, c+j+1,   r+i, 0)];
			mat7 = (t->data)[offset(t, cur_image2, c+j,   r+i+1, 0)];
			mat8 = (t->data)[offset(t, cur_image2, c+j+1, r+i+1, 0)];

			fil1 = fil_bin_w[f][i  ][j  ];
			fil2 = fil_bin_w[f][i  ][j+1];
			fil3 = fil_bin_w[f][i+1][j  ];
			fil4 = fil_bin_w[f][i+1][j+1];

			if (fil1 == 1)
			{

				conv_val1 += mat1;
				conv_val5 += mat5;
			}
			else
			{
				conv_val1 -= mat1;
				conv_val5 -= mat5;
			}

			if (fil2 == 1)
			{

				conv_val2 += mat2;
				conv_val6 += mat6;
			}
			else
			{
				conv_val2 -= mat2;
				conv_val6 -= mat6;
			}

			if (fil3 == 1)
			{

				conv_val3 += mat3;
				conv_val7 += mat7;
			}
			else
			{
				conv_val3 -= mat3;
				conv_val7 -= mat7;
			}

			if (fil4 == 1)
			{

				conv_val4 += mat4;
				conv_val8 += mat8;
			}
			else
			{
				conv_val4 -= mat4;
				conv_val8 -= mat8;
			}
		}

		// left over 1 element in the current row, at the end
		for (; j < FIL_COLS; ++j)
		{
			INCREMENT_FLOPS(2)

			if (fil_bin_w[f][i][j] == 1)
			{
				conv_img1_rem += (t->data)[offset(t, cur_image1, c+j, r+i, 0)];
				conv_img2_rem += (t->data)[offset(t, cur_image2, c+j, r+i, 0)];
			}
			else
			{
				conv_img1_rem -= (t->data)[offset(t, cur_image1, c+j, r+i, 0)];
				conv_img2_rem -= (t->data)[offset(t, cur_image2, c+j, r+i, 0)];
			}
		}
	}

	INCREMENT_FLOPS(8)
	conv_img1 = conv_val1 + conv_val2 + conv_val3 + conv_val4 + conv_img1_rem;
	conv_img2 = conv_val5 + conv_val6 + conv_val7 + conv_val8 + conv_img2_rem;


	// left over row, at the end
	for (; i < FIL_ROWS; ++i)
	{
		for (j = 0; j < FIL_COLS; ++j)
		{
			INCREMENT_FLOPS(2)

			if (fil_bin_w[f][i][j] == 1)
			{
				conv_img1_last += (t->data)[offset(t, cur_image1, c+j, r+i, 0)];
				conv_img2_last += (t->data)[offset(t, cur_image2, c+j, r+i, 0)];
			}
			else
			{
				conv_img1_last -= (t->data)[offset(t, cur_image1, c+j, r+i, 0)];
				conv_img2_last -= (t->data)[offset(t, cur_image2, c+j, r+i, 0)];
			}
		}
	}

	INCREMENT_FLOPS(10)
	conv_img1 += conv_img1_last;
	conv_img2 += conv_img2_last;

	conv_img1 *= alphas[f];
	conv_img2 *= alphas[f];

	

	conv_img1 += bias;
	conv_img2 += bias;

	// applying ReLU
	if (conv_img1 < 0.0)
	{
		conv_img1 = 0.0;
	}

	if (conv_img2 < 0.0)
	{
		conv_img2 = 0.0;
	}

	(conv_t->data)[offset(conv_t,b  ,c,r,f)] = conv_img1;
	(conv_t->data)[offset(conv_t,b+1,c,r,f)] = conv_img2;
}

void xnor_convolution(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], 
						tensor* conv_t, int batch_size, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
						double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[])
{
	for (int b = 0; b < batch_size; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					(conv_t->data)[offset(conv_t,b,j,i,f)] = xnor_convolve(bin_input_images, betas, i, j, b, 
																fil_bin_w, alphas, fil_b, f);
				}
			}
			
		}
	}
}

void initialize_filters(tensor* fil_w, tensor* fil_b) 
{
    srand(time(NULL));

    for(int k = 0; k < NUM_FILS; k++){
        (fil_b->data)[offset(fil_b, k, 0, 0, 0)] = 0.0;
        for(int i = 0; i < FIL_ROWS; ++i){
            for(int j = 0; j < FIL_COLS; ++j){
                for(int l = 0; l < FIL_DEPTH; ++l){
                    int r = rand();
                    double ran = ((double)rand())/RAND_MAX;
                    if (r%2 == 0){
                        (fil_w->data)[offset(fil_w, k, j, i, l)] = -ran;
                    }else{
                        (fil_w->data)[offset(fil_w, k, j, i, l)] = ran;
                    }
                }
            }
        }
    }
}

void print_filters(tensor* fil_w, tensor* fil_b)
{
	for (int k = 0; k < NUM_FILS; ++k)
	{
		printf("k=%d, bias=%f\nweights:\n", k, (fil_b->data)[offset(fil_b, k, 0, 0, 0)]);
		for (int i = 0; i < FIL_ROWS; ++i)
		{
			for (int j = 0; j < FIL_COLS; ++j)
			{
				printf("%.3f, ", (fil_w->data)[offset(fil_w, k, j, i, 0)]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void print_bin_filters(int bin_fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS])
{
	for (int k = 0; k < NUM_FILS; ++k)
	{
		printf("k=%d, alpha=%f\nweights:\n", k, alphas[k]);
		for (int i = 0; i < FIL_ROWS; ++i)
		{
			for (int j = 0; j < FIL_COLS; ++j)
			{
				printf("%3d, ", bin_fil_w[k][i][j]);				
			}
			printf("\n");
		}
		printf("\n");
	}
}

// Tested!!
double convolve(tensor* t, int r, int c, int image_num, tensor* fil_w, tensor* fil_b, int f, int shuffle_index[])
{

	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		for (int j = 0; j < FIL_COLS; ++j)
		{
            for (int k = 0; k < FIL_DEPTH; ++k){
			    conv_val += (fil_w->data)[offset(fil_w, f, j, i, k)] * (t->data)[offset(t, shuffle_index[image_num], c+j, r+i, k)];
            }
		}
	}

	conv_val += (fil_b->data)[offset(fil_b, f, 0, 0, 0)];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}

double xnor_convolve(int t[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV],
						int r, int c, int batch, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
						double alphas[NUM_FILS], tensor fil_b, int f)
{
	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		for (int j = 0; j < FIL_COLS; ++j)
		{

			// XNOR operation
			//conv_val += ( t[batch][i+r][j+c] == fil_bin_w[f][i][j] );

			conv_val += ( t[batch][i+r][j+c] * fil_bin_w[f][i][j] );
		}
	}

	conv_val *= alphas[f]*betas[batch][r][c];

	conv_val += fil_b.data[offset(&fil_b, f, 0, 0, 0)];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}