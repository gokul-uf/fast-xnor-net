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

/*// No loop unrolling
void bin_convolution(tensor* input_t, tensor* conv_t, int batch_size,
	int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[])
{
	for (int b = 0; b < batch_size; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					(conv_t->data)[ind_conv_out(b, f, i, j)] = bin_convolve(input_t, i, j, b+base, 
																fil_bin_w, alphas, fil_b, f, shuffle_index);
				}
			}
			
		}
	}
}

// No loop unrolling
double bin_convolve(tensor* t, int r, int c, int image_num, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS],
	tensor fil_b, int f, int shuffle_index[])
{
	int cur_image = shuffle_index[image_num];
	double conv_val = 0.0;
	double input_pixel;

	for (int i = 0; i < FIL_ROWS; ++i)
	{
		for (int j = 0; j < FIL_COLS; ++j)
		{
			INCREMENT_FLOPS(1)

			input_pixel = (t->data)[ind_input_img(cur_image, r+i, c+j)];

			if (fil_bin_w[f][i][j] == 1)
			{
				conv_val += input_pixel;
			}
			else
			{
				conv_val -= input_pixel;
			}
		}
	}

	INCREMENT_FLOPS(3)

	conv_val *= alphas[f];

	conv_val += fil_b.data[f];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}*/

// No unrolling
/*double xnor_convolve(int t[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV],
						int r, int c, int batch, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
						double alphas[NUM_FILS], tensor fil_b, int f)
{
	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		for (int j = 0; j < FIL_COLS; ++j)
		{
			INCREMENT_FLOPS(2)

			// XNOR operation
			//conv_val += ( t[batch][i+r][j+c] == fil_bin_w[f][i][j] );

			conv_val += ( t[batch][i+r][j+c] * fil_bin_w[f][i][j] );
		}
	}

	INCREMENT_FLOPS(4)

	conv_val *= alphas[f]*betas[batch][r][c];

	conv_val += fil_b.data[f];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}*/

// loop on number of filters unrolled
/*void bin_convolution(tensor* input_t, tensor* conv_t, int batch_size,
	int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[]){

	double conv_val0, conv_val1, conv_val2;
	double input_pixel;

	int cur_image;

	double alpha_0 = alphas[0];
	double alpha_1 = alphas[1];
	double alpha_2 = alphas[2];

	double bias_0 = fil_b.data[0];
	double bias_1 = fil_b.data[1];
	double bias_2 = fil_b.data[2];

	for (int b = 0; b < batch_size; ++b)
	{
		cur_image = shuffle_index[b+base];
		for (int r = 0; r < N_ROWS_CONV; ++r)
		{
			for (int c = 0; c < N_ROWS_CONV; ++c)
			{
				conv_val0 = 0.0, conv_val1 = 0.0, conv_val2 = 0.0;

				for (int i = 0; i < FIL_ROWS; ++i)
				{

					for (int j = 0; j < FIL_COLS; ++j)
					{
						input_pixel = (input_t->data)[ind_input_img(cur_image, r+i, c+j)];

						INCREMENT_FLOPS(3)
						// filter 0
						if (fil_bin_w[0][i][j] == 1)
						{
							conv_val0 += input_pixel;
						}
						else
						{
							conv_val0 -= input_pixel;
						}

						// filter 1
						if (fil_bin_w[1][i][j] == 1)
						{
							conv_val1 += input_pixel;
						}
						else
						{
							conv_val1 -= input_pixel;
						}

						// filter 2
						if (fil_bin_w[2][i][j] == 1)
						{
							conv_val2 += input_pixel;
						}
						else
						{
							conv_val2 -= input_pixel;
						}
					}
				}
			
				INCREMENT_FLOPS(12)

				conv_val0 *= alpha_0;
				conv_val0 += bias_0;

				conv_val1 *= alpha_1;
				conv_val1 += bias_1;

				conv_val2 *= alpha_2;
				conv_val2 += bias_2;

				// applying ReLU
				if (conv_val0 < 0.0)
				{
					conv_val0 = 0.0;
				}

				if (conv_val1 < 0.0)
				{
					conv_val1 = 0.0;
				}

				if (conv_val2 < 0.0)
				{
					conv_val2 = 0.0;
				}
			
				(conv_t->data)[ind_conv_out(b, 0, r, c)] = conv_val0;
				(conv_t->data)[ind_conv_out(b, 1, r, c)] = conv_val1;
				(conv_t->data)[ind_conv_out(b, 2, r, c)] = conv_val2;
			}
		}
	}
}*/

// convolution and pooling together
void bin_convolve_pool(tensor* input_t, tensor* conv_t, tensor* pool_t, int batch_size,
	int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[],
	int pool_index_i[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double conv_val_r0c0_f0, conv_val_r0c0_f1, conv_val_r0c0_f2;
	double conv_val_r0c1_f0, conv_val_r0c1_f1, conv_val_r0c1_f2;
	double conv_val_r1c0_f0, conv_val_r1c0_f1, conv_val_r1c0_f2;
	double conv_val_r1c1_f0, conv_val_r1c1_f1, conv_val_r1c1_f2;

	double input_pixel0, input_pixel1, input_pixel2, input_pixel3;

	int cur_image;

	double alpha_f0 = alphas[0];
	double alpha_f1 = alphas[1];
	double alpha_f2 = alphas[2];

	double bias_f0 = fil_b.data[0];
	double bias_f1 = fil_b.data[1];
	double bias_f2 = fil_b.data[2];

	double max_1_f0,   max_2_f0,     max_f0;
	int  max_1_i_f0, max_1_j_f0, max_2_i_f0, max_2_j_f0, max_i_f0, max_j_f0;

	double max_1_f1,   max_2_f1,     max_f1;
	int  max_1_i_f1, max_1_j_f1, max_2_i_f1, max_2_j_f1, max_i_f1, max_j_f1;

	double max_1_f2,   max_2_f2,    max_f2;
	int  max_1_i_f2, max_1_j_f2, max_2_i_f2, max_2_j_f2, max_i_f2, max_j_f2;

	int pool_i, pool_j;


	for (int b = 0; b < batch_size; ++b)
	{
		cur_image = shuffle_index[b+base];

		// Unroll r and c loops by 2 so that max pooling can be merged with convolution
		for (int r = 0, pool_i=0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
		{
			for (int c = 0, pool_j=0; c+1 < N_ROWS_CONV; c=c+2, ++pool_j)
			{
				conv_val_r0c0_f0 = 0.0, conv_val_r0c0_f1 = 0.0, conv_val_r0c0_f2 = 0.0;
				conv_val_r0c1_f0 = 0.0, conv_val_r0c1_f1 = 0.0, conv_val_r0c1_f2 = 0.0;
				conv_val_r1c0_f0 = 0.0, conv_val_r1c0_f1 = 0.0, conv_val_r1c0_f2 = 0.0;
				conv_val_r1c1_f0 = 0.0, conv_val_r1c1_f1 = 0.0, conv_val_r1c1_f2 = 0.0;

				for (int i = 0; i < FIL_ROWS; ++i)
				{

					for (int j = 0; j < FIL_COLS; ++j)
					{
						input_pixel0 = (input_t->data)[ind_input_img(cur_image, r+i  , c+j  )];
						input_pixel1 = (input_t->data)[ind_input_img(cur_image, r+i  , c+j+1)];
						input_pixel2 = (input_t->data)[ind_input_img(cur_image, r+i+1, c+j  )];
						input_pixel3 = (input_t->data)[ind_input_img(cur_image, r+i+1, c+j+1)];

						INCREMENT_FLOPS(12)
						// --------------------------------------------filter 0-------------------------------------
						if (fil_bin_w[0][i][j] == 1)
						{
							conv_val_r0c0_f0 += input_pixel0;
							conv_val_r0c1_f0 += input_pixel1;
							conv_val_r1c0_f0 += input_pixel2;
							conv_val_r1c1_f0 += input_pixel3;
						}
						else
						{
							conv_val_r0c0_f0 -= input_pixel0;
							conv_val_r0c1_f0 -= input_pixel1;
							conv_val_r1c0_f0 -= input_pixel2;
							conv_val_r1c1_f0 -= input_pixel3;
						}

						// --------------------------------------------filter 1-----------------------------------
						if (fil_bin_w[1][i][j] == 1)
						{
							conv_val_r0c0_f1 += input_pixel0;
							conv_val_r0c1_f1 += input_pixel1;
							conv_val_r1c0_f1 += input_pixel2;
							conv_val_r1c1_f1 += input_pixel3;
						}
						else
						{
							conv_val_r0c0_f1 -= input_pixel0;
							conv_val_r0c1_f1 -= input_pixel1;
							conv_val_r1c0_f1 -= input_pixel2;
							conv_val_r1c1_f1 -= input_pixel3;
						}

						// -------------------------------------------filter 2----------------------------------------------
						if (fil_bin_w[2][i][j] == 1)
						{
							conv_val_r0c0_f2 += input_pixel0;
							conv_val_r0c1_f2 += input_pixel1;
							conv_val_r1c0_f2 += input_pixel2;
							conv_val_r1c1_f2 += input_pixel3;
						}
						else
						{
							conv_val_r0c0_f2 -= input_pixel0;
							conv_val_r0c1_f2 -= input_pixel1;
							conv_val_r1c0_f2 -= input_pixel2;
							conv_val_r1c1_f2 -= input_pixel3;
						}

					}
				}
			
				INCREMENT_FLOPS(36)

				// -----------------------------------------------filter 0 ----------------------------------------------
				conv_val_r0c0_f0 *= alpha_f0;
				conv_val_r0c0_f0 +=  bias_f0;

				conv_val_r0c1_f0 *= alpha_f0;
				conv_val_r0c1_f0 +=  bias_f0;

				conv_val_r1c0_f0 *= alpha_f0;
				conv_val_r1c0_f0 +=  bias_f0;

				conv_val_r1c1_f0 *= alpha_f0;
				conv_val_r1c1_f0 +=  bias_f0;

				// -----------------------------------------------filter 1---------------------------------------------
				conv_val_r0c0_f1 *= alpha_f1;
				conv_val_r0c0_f1 +=  bias_f1;

				conv_val_r0c1_f1 *= alpha_f1;
				conv_val_r0c1_f1 +=  bias_f1;

				conv_val_r1c0_f1 *= alpha_f1;
				conv_val_r1c0_f1 +=  bias_f1;

				conv_val_r1c1_f1 *= alpha_f1;
				conv_val_r1c1_f1 +=  bias_f1;

				// -----------------------------------------------filter 2---------------------------------------------
				conv_val_r0c0_f2 *= alpha_f2;
				conv_val_r0c0_f2 +=  bias_f2;

				conv_val_r0c1_f2 *= alpha_f2;
				conv_val_r0c1_f2 +=  bias_f2;

				conv_val_r1c0_f2 *= alpha_f2;
				conv_val_r1c0_f2 +=  bias_f2;

				conv_val_r1c1_f2 *= alpha_f2;
				conv_val_r1c1_f2 +=  bias_f2;

				// applying ReLU
				// -------------------------------------------filter 0------------------------------------------------
				if (conv_val_r0c0_f0 < 0.0)
				{
					conv_val_r0c0_f0 = 0.0;
				}

				if (conv_val_r0c1_f0 < 0.0)
				{
					conv_val_r0c1_f0 = 0.0;
				}

				if (conv_val_r1c0_f0 < 0.0)
				{
					conv_val_r1c0_f0 = 0.0;
				}

				if (conv_val_r1c1_f0 < 0.0)
				{
					conv_val_r1c1_f0 = 0.0;
				}

				// -------------------------------------------filter 1------------------------------------------------
				if (conv_val_r0c0_f1 < 0.0)
				{
					conv_val_r0c0_f1 = 0.0;
				}

				if (conv_val_r0c1_f1 < 0.0)
				{
					conv_val_r0c1_f1 = 0.0;
				}

				if (conv_val_r1c0_f1 < 0.0)
				{
					conv_val_r1c0_f1 = 0.0;
				}

				if (conv_val_r1c1_f1 < 0.0)
				{
					conv_val_r1c1_f1 = 0.0;
				}

				// -------------------------------------------filter 2------------------------------------------------
				if (conv_val_r0c0_f2 < 0.0)
				{
					conv_val_r0c0_f2 = 0.0;
				}

				if (conv_val_r0c1_f2 < 0.0)
				{
					conv_val_r0c1_f2 = 0.0;
				}

				if (conv_val_r1c0_f2 < 0.0)
				{
					conv_val_r1c0_f2 = 0.0;
				}

				if (conv_val_r1c1_f2 < 0.0)
				{
					conv_val_r1c1_f2 = 0.0;
				}

				(conv_t->data)[ind_conv_out(b, 0, r  , c  )] = conv_val_r0c0_f0;
				(conv_t->data)[ind_conv_out(b, 0, r  , c+1)] = conv_val_r0c1_f0;
				(conv_t->data)[ind_conv_out(b, 0, r+1, c  )] = conv_val_r1c0_f0;
				(conv_t->data)[ind_conv_out(b, 0, r+1, c+1)] = conv_val_r1c1_f0;

				(conv_t->data)[ind_conv_out(b, 1, r  , c  )] = conv_val_r0c0_f1;
				(conv_t->data)[ind_conv_out(b, 1, r  , c+1)] = conv_val_r0c1_f1;
				(conv_t->data)[ind_conv_out(b, 1, r+1, c  )] = conv_val_r1c0_f1;
				(conv_t->data)[ind_conv_out(b, 1, r+1, c+1)] = conv_val_r1c1_f1;

				(conv_t->data)[ind_conv_out(b, 2, r  , c  )] = conv_val_r0c0_f2;
				(conv_t->data)[ind_conv_out(b, 2, r  , c+1)] = conv_val_r0c1_f2;
				(conv_t->data)[ind_conv_out(b, 2, r+1, c  )] = conv_val_r1c0_f2;
				(conv_t->data)[ind_conv_out(b, 2, r+1, c+1)] = conv_val_r1c1_f2;

				// --------------------------------------------Max Pooling-------------------------------------
				INCREMENT_FLOPS(9)

				
				// -------------------------------------------Filter 0----------------------------------------
				if (conv_val_r0c0_f0 > conv_val_r0c1_f0)
				{
					  max_1_f0 = conv_val_r0c0_f0;
					max_1_i_f0 = r;
					max_1_j_f0 = c;
				}
				else
				{
					  max_1_f0 = conv_val_r0c1_f0;
					max_1_i_f0 = r;
					max_1_j_f0 = c+1;
				}

				if (conv_val_r1c0_f0 > conv_val_r1c1_f0)
				{
					  max_2_f0 = conv_val_r1c0_f0;
					max_2_i_f0 = r+1;
					max_2_j_f0 = c;
				}
				else
				{
					  max_2_f0 = conv_val_r1c1_f0;
					max_2_i_f0 = r+1;
					max_2_j_f0 = c+1;
				}

				if (max_1_f0 > max_2_f0)
				{
					  max_f0 =   max_1_f0;
					max_i_f0 = max_1_i_f0;
					max_j_f0 = max_1_j_f0;
				}
				else
				{
					  max_f0 =   max_2_f0;
					max_i_f0 = max_2_i_f0;
					max_j_f0 = max_2_j_f0;
				}

				(pool_t->data)[ind_pool_out(b, 0, pool_i, pool_j)] = max_f0;
				pool_index_i[b][0][pool_i][pool_j] = max_i_f0;
				pool_index_j[b][0][pool_i][pool_j] = max_j_f0;


				// -------------------------------------------Filter 1----------------------------------------
				if (conv_val_r0c0_f1 > conv_val_r0c1_f1)
				{
					  max_1_f1 = conv_val_r0c0_f1;
					max_1_i_f1 = r;
					max_1_j_f1 = c;
				}
				else
				{
					  max_1_f1 = conv_val_r0c1_f1;
					max_1_i_f1 = r;
					max_1_j_f1 = c+1;
				}

				if (conv_val_r1c0_f1 > conv_val_r1c1_f1)
				{
					  max_2_f1 = conv_val_r1c0_f1;
					max_2_i_f1 = r+1;
					max_2_j_f1 = c;
				}
				else
				{
					  max_2_f1 = conv_val_r1c1_f1;
					max_2_i_f1 = r+1;
					max_2_j_f1 = c+1;
				}

				if (max_1_f1 > max_2_f1)
				{
					  max_f1 =   max_1_f1;
					max_i_f1 = max_1_i_f1;
					max_j_f1 = max_1_j_f1;
				}
				else
				{
					  max_f1 =   max_2_f1;
					max_i_f1 = max_2_i_f1;
					max_j_f1 = max_2_j_f1;
				}

				(pool_t->data)[ind_pool_out(b, 1, pool_i, pool_j)] = max_f1;
				pool_index_i[b][1][pool_i][pool_j] = max_i_f1;
				pool_index_j[b][1][pool_i][pool_j] = max_j_f1;

				// -------------------------------------------Filter 2----------------------------------------
				if (conv_val_r0c0_f2 > conv_val_r0c1_f2)
				{
					  max_1_f2 = conv_val_r0c0_f2;
					max_1_i_f2 = r;
					max_1_j_f2 = c;
				}
				else
				{
					  max_1_f2 = conv_val_r0c1_f2;
					max_1_i_f2 = r;
					max_1_j_f2 = c+1;
				}

				if (conv_val_r1c0_f2 > conv_val_r1c1_f2)
				{
					  max_2_f2 = conv_val_r1c0_f2;
					max_2_i_f2 = r+1;
					max_2_j_f2 = c;
				}
				else
				{
					  max_2_f2 = conv_val_r1c1_f2;
					max_2_i_f2 = r+1;
					max_2_j_f2 = c+1;
				}

				if (max_1_f2 > max_2_f2)
				{
					  max_f2 =   max_1_f2;
					max_i_f2 = max_1_i_f2;
					max_j_f2 = max_1_j_f2;
				}
				else
				{
					  max_f2 =   max_2_f2;
					max_i_f2 = max_2_i_f2;
					max_j_f2 = max_2_j_f2;
				}

				(pool_t->data)[ind_pool_out(b, 2, pool_i, pool_j)] = max_f2;
				pool_index_i[b][2][pool_i][pool_j] = max_i_f2;
				pool_index_j[b][2][pool_i][pool_j] = max_j_f2;
			}
		}
	}
}


// loop on conv rows and cols unrolled by 2, max-pooling done
/*void xnor_convolve_pool(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], 
					tensor* conv_t, int batch_size, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
					double alphas[NUM_FILS], tensor fil_b, tensor* pool_t,
					int pool_index_i[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double conv_val0, conv_val1, conv_val2, conv_val3;
	double alpha, bias;
	double beta0, beta1, beta2, beta3;
	double weight;

	double prev1, prev2, curr1, curr2;

	double input_pixel0;
	double input_pixel1;
	double input_pixel2;
	double input_pixel3;

	int pool_i, pool_j;
	int max_pool_1, max_pool_2, max_pool;
	int    ind_1_i,    ind_2_i,    ind_i;
	int    ind_1_j,    ind_2_j,    ind_j;

	for (int b = 0; b < batch_size; ++b)
	{

		for (int f = 0; f < NUM_FILS; ++f)
		{

			alpha = alphas[f];
			bias = fil_b.data[f];

			for (int r = 0, pool_i = 0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
			{

				for (int c = 0, pool_j = 0; c+1 < N_COLS_CONV; c=c+2, ++pool_j)
				{

					beta0 = betas[b][r  ][c  ];
					beta1 = betas[b][r  ][c+1];
					beta2 = betas[b][r+1][c  ];
					beta3 = betas[b][r+1][c+1];

					conv_val0 = 0.0;
					conv_val1 = 0.0;
					conv_val2 = 0.0;
					conv_val3 = 0.0;

					for (int i = 0; i < FIL_ROWS; ++i)
					{

						prev1 = bin_input_images[b][i+r  ][0+c  ];
						prev2 = bin_input_images[b][i+r+1][0+c  ];

						for (int j = 0; j < FIL_COLS; ++j)
						{

							INCREMENT_FLOPS(8)

							weight = fil_bin_w[f][i][j];

							input_pixel0 = prev1;
							input_pixel1 = bin_input_images[b][i+r  ][j+c+1];
							input_pixel2 = prev2;
							input_pixel3 = bin_input_images[b][i+r+1][j+c+1];

							// XNOR operation
							//conv_val += ( bin_input_images[b][i+r][j+c] == fil_bin_w[f][i][j] );

							conv_val0 += ( input_pixel0 * weight );
							conv_val1 += ( input_pixel1 * weight );
							conv_val2 += ( input_pixel2 * weight );
							conv_val3 += ( input_pixel3 * weight );

							prev1 = input_pixel1;
							prev2 = input_pixel3;
						}
					}

					INCREMENT_FLOPS(16)

					conv_val0 *= alpha * beta0;
					conv_val1 *= alpha * beta1;
					conv_val2 *= alpha * beta2;
					conv_val3 *= alpha * beta3;

					conv_val0 += bias;
					conv_val1 += bias;
					conv_val2 += bias;
					conv_val3 += bias;

					// applying ReLU
					if (conv_val0 < 0.0)
					{
						conv_val0 = 0.0;
					}

					if (conv_val1 < 0.0)
					{
						conv_val1 = 0.0;
					}

					if (conv_val2 < 0.0)
					{
						conv_val2 = 0.0;
					}

					if (conv_val3 < 0.0)
					{
						conv_val3 = 0.0;
					}

					(conv_t->data)[ind_conv_out(b, f, r  , c  )] = conv_val0;
					(conv_t->data)[ind_conv_out(b, f, r  , c+1)] = conv_val1;
					(conv_t->data)[ind_conv_out(b, f, r+1, c  )] = conv_val2;
					(conv_t->data)[ind_conv_out(b, f, r+1, c+1)] = conv_val3;


					// ----------------------------------------------Max pooling---------------------------------------

					if (conv_val0 > conv_val1)
					{
						max_pool_1 = conv_val0;
						   ind_1_i = r;
						   ind_1_j = c;
					}
					else
					{
						max_pool_1 = conv_val1;
						   ind_1_i = r  ;
						   ind_1_j = c+1;
					}

					if (conv_val2 > conv_val3)
					{
						max_pool_2 = conv_val2;
						   ind_2_i = r+1;
						   ind_2_j = c  ;
					}
					else
					{
						max_pool_2 = conv_val3;
						   ind_2_i = r+1;
						   ind_2_j = c+1;
					}

					if (max_pool_1 > max_pool_2)
					{
						max_pool = max_pool_1;
						   ind_i = ind_1_i;
						   ind_j = ind_1_j;
					}
					else
					{
						max_pool = max_pool_2;
						   ind_i = ind_2_i;
						   ind_j = ind_2_j;
					}

					(pool_t->data)[ind_pool_out(b, f, pool_i, pool_j)] = max_pool;

					pool_index_i[b][f][pool_i][pool_j] = ind_i;
					pool_index_j[b][f][pool_i][pool_j] = ind_j;
				}
			}
			
		}
	}
}*/

// loop on conv rows and cols unrolled by 2, max-pooling done, pool indexes saved; loop on number of filters unrolled
void xnor_convolve_pool(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], 
					tensor* conv_t, int batch_size, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
					double alphas[NUM_FILS], tensor fil_b, tensor* pool_t, 
					int pool_index_i[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double conv_val0, conv_val1, conv_val2, conv_val3;
	double alpha, bias;
	double beta0, beta1, beta2, beta3;
	double weight;

	double prev1, prev2, curr1, curr2;

	double input_pixel0;
	double input_pixel1;
	double input_pixel2;
	double input_pixel3;

	int pool_i, pool_j;
	int max_pool_1, max_pool_2, max_pool;
	int    ind_1_i,    ind_2_i,    ind_i;
	int    ind_1_j,    ind_2_j,    ind_j;

	int max_pool_1_f0, max_pool_2_f0, max_pool_f0;
	int    ind_1_i_f0,    ind_2_i_f0,    ind_i_f0;
	int    ind_1_j_f0,    ind_2_j_f0,    ind_j_f0;

	int max_pool_1_f1, max_pool_2_f1, max_pool_f1;
	int    ind_1_i_f1,    ind_2_i_f1,    ind_i_f1;
	int    ind_1_j_f1,    ind_2_j_f1,    ind_j_f1;

	int max_pool_1_f2, max_pool_2_f2, max_pool_f2;
	int    ind_1_i_f2,    ind_2_i_f2,    ind_i_f2;
	int    ind_1_j_f2,    ind_2_j_f2,    ind_j_f2;

	double alpha_f0;
	double  bias_f0;
	double alpha_f1;
	double  bias_f1;
	double alpha_f2;
	double  bias_f2;

	double conv_val0_f0;
	double conv_val1_f0;
	double conv_val2_f0;
	double conv_val3_f0;
	double conv_val0_f1;
	double conv_val1_f1;
	double conv_val2_f1;
	double conv_val3_f1;
	double conv_val0_f2;
	double conv_val1_f2;
	double conv_val2_f2;
	double conv_val3_f2;

	double weight_f0;
	double weight_f1;
	double weight_f2;

	int f;

	for (int b = 0; b < batch_size; ++b)
	{

		for (f = 0; f+2 < NUM_FILS; f=f+3)
		{

			alpha_f0 = alphas[f];
			 bias_f0 = fil_b.data[f];

		 	alpha_f1 = alphas[f+1];
			 bias_f1 = fil_b.data[f+1];

		 	alpha_f2 = alphas[f+2];
			 bias_f2 = fil_b.data[f+2];

			for (int r = 0, pool_i = 0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
			{

				for (int c = 0, pool_j = 0; c+1 < N_COLS_CONV; c=c+2, ++pool_j)
				{

					beta0 = betas[b][r  ][c  ];
					beta1 = betas[b][r  ][c+1];
					beta2 = betas[b][r+1][c  ];
					beta3 = betas[b][r+1][c+1];

					conv_val0_f0 = 0.0;
					conv_val1_f0 = 0.0;
					conv_val2_f0 = 0.0;
					conv_val3_f0 = 0.0;

					conv_val0_f1 = 0.0;
					conv_val1_f1 = 0.0;
					conv_val2_f1 = 0.0;
					conv_val3_f1 = 0.0;

					conv_val0_f2 = 0.0;
					conv_val1_f2 = 0.0;
					conv_val2_f2 = 0.0;
					conv_val3_f2 = 0.0;

					for (int i = 0; i < FIL_ROWS; ++i)
					{

						prev1 = bin_input_images[b][i+r  ][0+c  ];
						prev2 = bin_input_images[b][i+r+1][0+c  ];

						for (int j = 0; j < FIL_COLS; ++j)
						{

							INCREMENT_FLOPS(24)

							// load filters
							weight_f0 = fil_bin_w[f  ][i][j];
							weight_f1 = fil_bin_w[f+1][i][j];
							weight_f2 = fil_bin_w[f+2][i][j];

							// load pixels from inputs
							input_pixel0 = prev1;
							input_pixel1 = bin_input_images[b][i+r  ][j+c+1];
							input_pixel2 = prev2;
							input_pixel3 = bin_input_images[b][i+r+1][j+c+1];

							// XNOR operation
							//conv_val += ( bin_input_images[b][i+r][j+c] == fil_bin_w[f][i][j] );

							// do element wise product
							conv_val0_f0 += ( input_pixel0 * weight_f0 );
							conv_val1_f0 += ( input_pixel1 * weight_f0 );
							conv_val2_f0 += ( input_pixel2 * weight_f0 );
							conv_val3_f0 += ( input_pixel3 * weight_f0 );

							conv_val0_f1 += ( input_pixel0 * weight_f1 );
							conv_val1_f1 += ( input_pixel1 * weight_f1 );
							conv_val2_f1 += ( input_pixel2 * weight_f1 );
							conv_val3_f1 += ( input_pixel3 * weight_f1 );

							conv_val0_f2 += ( input_pixel0 * weight_f2 );
							conv_val1_f2 += ( input_pixel1 * weight_f2 );
							conv_val2_f2 += ( input_pixel2 * weight_f2 );
							conv_val3_f2 += ( input_pixel3 * weight_f2 );

							// store loaded pixels for reuse in next iteration
							prev1 = input_pixel1;
							prev2 = input_pixel3;
						}
					}

					INCREMENT_FLOPS(48)

					// ----------------------------------Filter 0------------------------------------------------
					conv_val0_f0 *= alpha_f0 * beta0;
					conv_val1_f0 *= alpha_f0 * beta1;
					conv_val2_f0 *= alpha_f0 * beta2;
					conv_val3_f0 *= alpha_f0 * beta3;

					conv_val0_f0 += bias_f0;
					conv_val1_f0 += bias_f0;
					conv_val2_f0 += bias_f0;
					conv_val3_f0 += bias_f0;

					// ----------------------------------Filter 1------------------------------------------------
					conv_val0_f1 *= alpha_f1 * beta0;
					conv_val1_f1 *= alpha_f1 * beta1;
					conv_val2_f1 *= alpha_f1 * beta2;
					conv_val3_f1 *= alpha_f1 * beta3;

					conv_val0_f1 += bias_f1;
					conv_val1_f1 += bias_f1;
					conv_val2_f1 += bias_f1;
					conv_val3_f1 += bias_f1;

					// ----------------------------------Filter 2------------------------------------------------
					conv_val0_f2 *= alpha_f2 * beta0;
					conv_val1_f2 *= alpha_f2 * beta1;
					conv_val2_f2 *= alpha_f2 * beta2;
					conv_val3_f2 *= alpha_f2 * beta3;

					conv_val0_f2 += bias_f2;
					conv_val1_f2 += bias_f2;
					conv_val2_f2 += bias_f2;
					conv_val3_f2 += bias_f2;


					// ------------------------------------applying ReLU------------------------------------------
					// ------------------------------------Filter 0-----------------------------------------------
					if (conv_val0_f0 < 0.0)
					{
						conv_val0_f0 = 0.0;
					}

					if (conv_val1_f0 < 0.0)
					{
						conv_val1_f0 = 0.0;
					}

					if (conv_val2_f0 < 0.0)
					{
						conv_val2_f0 = 0.0;
					}

					if (conv_val3_f0 < 0.0)
					{
						conv_val3_f0 = 0.0;
					}

					// ------------------------------------Filter 1-----------------------------------------------
					if (conv_val0_f1 < 0.0)
					{
						conv_val0_f1 = 0.0;
					}

					if (conv_val1_f1 < 0.0)
					{
						conv_val1_f1 = 0.0;
					}

					if (conv_val2_f1 < 0.0)
					{
						conv_val2_f1 = 0.0;
					}

					if (conv_val3_f1 < 0.0)
					{
						conv_val3_f1 = 0.0;
					}

					// ------------------------------------Filter 2-----------------------------------------------
					if (conv_val0_f2 < 0.0)
					{
						conv_val0_f2 = 0.0;
					}

					if (conv_val1_f2 < 0.0)
					{
						conv_val1_f2 = 0.0;
					}

					if (conv_val2_f2 < 0.0)
					{
						conv_val2_f2 = 0.0;
					}

					if (conv_val3_f2 < 0.0)
					{
						conv_val3_f2 = 0.0;
					}

					(conv_t->data)[ind_conv_out(b, f  , r  , c  )] = conv_val0_f0;
					(conv_t->data)[ind_conv_out(b, f  , r  , c+1)] = conv_val1_f0;
					(conv_t->data)[ind_conv_out(b, f  , r+1, c  )] = conv_val2_f0;
					(conv_t->data)[ind_conv_out(b, f  , r+1, c+1)] = conv_val3_f0;

					(conv_t->data)[ind_conv_out(b, f+1, r  , c  )] = conv_val0_f1;
					(conv_t->data)[ind_conv_out(b, f+1, r  , c+1)] = conv_val1_f1;
					(conv_t->data)[ind_conv_out(b, f+1, r+1, c  )] = conv_val2_f1;
					(conv_t->data)[ind_conv_out(b, f+1, r+1, c+1)] = conv_val3_f1;

					(conv_t->data)[ind_conv_out(b, f+2, r  , c  )] = conv_val0_f2;
					(conv_t->data)[ind_conv_out(b, f+2, r  , c+1)] = conv_val1_f2;
					(conv_t->data)[ind_conv_out(b, f+2, r+1, c  )] = conv_val2_f2;
					(conv_t->data)[ind_conv_out(b, f+2, r+1, c+1)] = conv_val3_f2;

					// ----------------------------------------------Max pooling---------------------------------------

					INCREMENT_FLOPS(9)

					// -----------------------------------------------Filter 0----------------------------------------
					if ( conv_val0_f0 > conv_val1_f0)
					{
						max_pool_1_f0 = conv_val0_f0;
						   ind_1_i_f0 = r;
						   ind_1_j_f0 = c;
					}
					else
					{
						max_pool_1_f0 = conv_val1_f0;
						   ind_1_i_f0 = r  ;
						   ind_1_j_f0 = c+1;
					}

					if (conv_val2_f0 > conv_val3_f0)
					{
						max_pool_2_f0 = conv_val2_f0;
						   ind_2_i_f0 = r+1;
						   ind_2_j_f0 = c  ;
					}
					else
					{
						max_pool_2_f0 = conv_val3_f0;
						   ind_2_i_f0 = r+1;
						   ind_2_j_f0 = c+1;
					}

					if (max_pool_1_f0 > max_pool_2_f0)
					{
						max_pool_f0 = max_pool_1_f0;
						   ind_i_f0 = ind_1_i_f0;
						   ind_j_f0 = ind_1_j_f0;
					}
					else
					{
						max_pool_f0 = max_pool_2_f0;
						   ind_i_f0 = ind_2_i_f0;
						   ind_j_f0 = ind_2_j_f0;
					}

					// -----------------------------------------------Filter 1----------------------------------------
					if (conv_val0_f1 > conv_val1_f1)
					{
						max_pool_1_f1 = conv_val0_f1;
						   ind_1_i_f1 = r;
						   ind_1_j_f1 = c;
					}
					else
					{
						max_pool_1_f1 = conv_val1_f1;
						   ind_1_i_f1 = r  ;
						   ind_1_j_f1 = c+1;
					}

					if (conv_val2_f1 > conv_val3_f1)
					{
						max_pool_2_f1 = conv_val2_f1;
						   ind_2_i_f1 = r+1;
						   ind_2_j_f1 = c  ;
					}
					else
					{
						max_pool_2_f1 = conv_val3_f1;
						   ind_2_i_f1 = r+1;
						   ind_2_j_f1 = c+1;
					}

					if (max_pool_1_f1 > max_pool_2_f1)
					{
						max_pool_f1 = max_pool_1_f1;
						   ind_i_f1 = ind_1_i_f1;
						   ind_j_f1 = ind_1_j_f1;
					}
					else
					{
						max_pool_f1 = max_pool_2_f1;
						   ind_i_f1 = ind_2_i_f1;
						   ind_j_f1 = ind_2_j_f1;
					}

					// -----------------------------------------------Filter 2----------------------------------------
					if (conv_val0_f2 > conv_val1_f2)
					{
						max_pool_1_f2 = conv_val0_f2;
						   ind_1_i_f2 = r;
						   ind_1_j_f2 = c;
					}
					else
					{
						max_pool_1_f2 = conv_val1_f2;
						   ind_1_i_f2 = r  ;
						   ind_1_j_f2 = c+1;
					}

					if (conv_val2_f2 > conv_val3_f2)
					{
						max_pool_2_f2 = conv_val2_f2;
						   ind_2_i_f2 = r+1;
						   ind_2_j_f2 = c  ;
					}
					else
					{
						max_pool_2_f2 = conv_val3_f2;
						   ind_2_i_f2 = r+1;
						   ind_2_j_f2 = c+1;
					}

					if (max_pool_1_f2 > max_pool_2_f2)
					{
						max_pool_f2 = max_pool_1_f2;
						   ind_i_f2 = ind_1_i_f2;
						   ind_j_f2 = ind_1_j_f2;
					}
					else
					{
						max_pool_f2 = max_pool_2_f2;
						   ind_i_f2 = ind_2_i_f2;
						   ind_j_f2 = ind_2_j_f2;
					}


					(pool_t->data)[ind_pool_out(b, f  , pool_i, pool_j)] = max_pool_f0;
					(pool_t->data)[ind_pool_out(b, f+1, pool_i, pool_j)] = max_pool_f1;
					(pool_t->data)[ind_pool_out(b, f+2, pool_i, pool_j)] = max_pool_f2;

					pool_index_i[b][f  ][pool_i][pool_j] = ind_i_f0;
					pool_index_j[b][f  ][pool_i][pool_j] = ind_j_f0;

					pool_index_i[b][f+1][pool_i][pool_j] = ind_i_f1;
					pool_index_j[b][f+1][pool_i][pool_j] = ind_j_f1;

					pool_index_i[b][f+2][pool_i][pool_j] = ind_i_f2;
					pool_index_j[b][f+2][pool_i][pool_j] = ind_j_f2;
				}
			}			
		}


		// if number of filters not divisible by 3
		for (; f < NUM_FILS; ++f)
		{

			alpha = alphas[f];
			bias = fil_b.data[f];

			for (int r = 0, pool_i = 0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
			{

				for (int c = 0, pool_j = 0; c+1 < N_COLS_CONV; c=c+2, ++pool_j)
				{

					beta0 = betas[b][r  ][c  ];
					beta1 = betas[b][r  ][c+1];
					beta2 = betas[b][r+1][c  ];
					beta3 = betas[b][r+1][c+1];

					conv_val0 = 0.0;
					conv_val1 = 0.0;
					conv_val2 = 0.0;
					conv_val3 = 0.0;

					for (int i = 0; i < FIL_ROWS; ++i)
					{

						prev1 = bin_input_images[b][i+r  ][0+c  ];
						prev2 = bin_input_images[b][i+r+1][0+c  ];

						for (int j = 0; j < FIL_COLS; ++j)
						{

							INCREMENT_FLOPS(8)

							weight = fil_bin_w[f][i][j];

							input_pixel0 = prev1;
							input_pixel1 = bin_input_images[b][i+r  ][j+c+1];
							input_pixel2 = prev2;
							input_pixel3 = bin_input_images[b][i+r+1][j+c+1];

							// XNOR operation
							//conv_val += ( bin_input_images[b][i+r][j+c] == fil_bin_w[f][i][j] );

							conv_val0 += ( input_pixel0 * weight );
							conv_val1 += ( input_pixel1 * weight );
							conv_val2 += ( input_pixel2 * weight );
							conv_val3 += ( input_pixel3 * weight );

							prev1 = input_pixel1;
							prev2 = input_pixel3;
						}
					}

					INCREMENT_FLOPS(16)

					conv_val0 *= alpha * beta0;
					conv_val1 *= alpha * beta1;
					conv_val2 *= alpha * beta2;
					conv_val3 *= alpha * beta3;

					conv_val0 += bias;
					conv_val1 += bias;
					conv_val2 += bias;
					conv_val3 += bias;

					// applying ReLU
					if (conv_val0 < 0.0)
					{
						conv_val0 = 0.0;
					}

					if (conv_val1 < 0.0)
					{
						conv_val1 = 0.0;
					}

					if (conv_val2 < 0.0)
					{
						conv_val2 = 0.0;
					}

					if (conv_val3 < 0.0)
					{
						conv_val3 = 0.0;
					}

					(conv_t->data)[ind_conv_out(b, f, r  , c  )] = conv_val0;
					(conv_t->data)[ind_conv_out(b, f, r  , c+1)] = conv_val1;
					(conv_t->data)[ind_conv_out(b, f, r+1, c  )] = conv_val2;
					(conv_t->data)[ind_conv_out(b, f, r+1, c+1)] = conv_val3;


					// ----------------------------------------------Max pooling---------------------------------------

					INCREMENT_FLOPS(3)

					if (conv_val0 > conv_val1)
					{
						max_pool_1 = conv_val0;
						   ind_1_i = r;
						   ind_1_j = c;
					}
					else
					{
						max_pool_1 = conv_val1;
						   ind_1_i = r  ;
						   ind_1_j = c+1;
					}

					if (conv_val2 > conv_val3)
					{
						max_pool_2 = conv_val2;
						   ind_2_i = r+1;
						   ind_2_j = c  ;
					}
					else
					{
						max_pool_2 = conv_val3;
						   ind_2_i = r+1;
						   ind_2_j = c+1;
					}

					if (max_pool_1 > max_pool_2)
					{
						max_pool = max_pool_1;
						   ind_i = ind_1_i;
						   ind_j = ind_1_j;
					}
					else
					{
						max_pool = max_pool_2;
						   ind_i = ind_2_i;
						   ind_j = ind_2_j;
					}

					(pool_t->data)[ind_pool_out(b, f, pool_i, pool_j)] = max_pool;

					pool_index_i[b][f][pool_i][pool_j] = ind_i;
					pool_index_j[b][f][pool_i][pool_j] = ind_j;
				}
			}			
		}
	}
}

// loop on conv rows and cols unrolled by 2, max-pooling done, pool indexes not saved
void xnor_convolve_pool_validation(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], 
						tensor* conv_t, int batch_size, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
						double alphas[NUM_FILS], tensor fil_b, tensor* pool_t)
{
	double conv_val0, conv_val1, conv_val2, conv_val3;
	double alpha, bias;
	double beta0, beta1, beta2, beta3;
	double weight;

	double prev1, prev2, curr1, curr2;

	double input_pixel0;
	double input_pixel1;
	double input_pixel2;
	double input_pixel3;

	int pool_i, pool_j;
	int max_pool_1, max_pool_2, max_pool;

	for (int b = 0; b < batch_size; ++b)
	{

		for (int f = 0; f < NUM_FILS; ++f)
		{

			alpha = alphas[f];
			bias = fil_b.data[f];

			for (int r = 0, pool_i = 0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
			{

				for (int c = 0, pool_j = 0; c+1 < N_COLS_CONV; c=c+2, ++pool_j)
				{

					beta0 = betas[b][r  ][c  ];
					beta1 = betas[b][r  ][c+1];
					beta2 = betas[b][r+1][c  ];
					beta3 = betas[b][r+1][c+1];

					conv_val0 = 0.0;
					conv_val1 = 0.0;
					conv_val2 = 0.0;
					conv_val3 = 0.0;

					for (int i = 0; i < FIL_ROWS; ++i)
					{

						prev1 = bin_input_images[b][i+r  ][0+c  ];
						prev2 = bin_input_images[b][i+r+1][0+c  ];

						for (int j = 0; j < FIL_COLS; ++j)
						{

							INCREMENT_FLOPS(8)

							weight = fil_bin_w[f][i][j];

							input_pixel0 = prev1;
							input_pixel1 = bin_input_images[b][i+r  ][j+c+1];
							input_pixel2 = prev2;
							input_pixel3 = bin_input_images[b][i+r+1][j+c+1];

							// XNOR operation
							//conv_val += ( bin_input_images[b][i+r][j+c] == fil_bin_w[f][i][j] );

							conv_val0 += ( input_pixel0 * weight );
							conv_val1 += ( input_pixel1 * weight );
							conv_val2 += ( input_pixel2 * weight );
							conv_val3 += ( input_pixel3 * weight );

							prev1 = input_pixel1;
							prev2 = input_pixel3;
						}
					}

					INCREMENT_FLOPS(16)

					conv_val0 *= alpha * beta0;
					conv_val1 *= alpha * beta1;
					conv_val2 *= alpha * beta2;
					conv_val3 *= alpha * beta3;

					conv_val0 += bias;
					conv_val1 += bias;
					conv_val2 += bias;
					conv_val3 += bias;

					// applying ReLU
					if (conv_val0 < 0.0)
					{
						conv_val0 = 0.0;
					}

					if (conv_val1 < 0.0)
					{
						conv_val1 = 0.0;
					}

					if (conv_val2 < 0.0)
					{
						conv_val2 = 0.0;
					}

					if (conv_val3 < 0.0)
					{
						conv_val3 = 0.0;
					}

					(conv_t->data)[ind_conv_out(b, f, r  , c  )] = conv_val0;
					(conv_t->data)[ind_conv_out(b, f, r  , c+1)] = conv_val1;
					(conv_t->data)[ind_conv_out(b, f, r+1, c  )] = conv_val2;
					(conv_t->data)[ind_conv_out(b, f, r+1, c+1)] = conv_val3;


					// ----------------------------------------------Max pooling---------------------------------------

					if (conv_val0 > conv_val1)
					{
						max_pool_1 = conv_val0;
					}
					else
					{
						max_pool_1 = conv_val1;
					}

					if (conv_val2 > conv_val3)
					{
						max_pool_2 = conv_val2;
					}
					else
					{
						max_pool_2 = conv_val3;
					}

					if (max_pool_1 > max_pool_2)
					{
						max_pool = max_pool_1;
					}
					else
					{
						max_pool = max_pool_2;
					}

					(pool_t->data)[ind_pool_out(b, f, pool_i, pool_j)] = max_pool;
				}
			}
			
		}
	}
}

// loop on conv rows and cols unrolled by 2
/*void xnor_convolution(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], 
						tensor* conv_t, int batch_size, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], 
						double alphas[NUM_FILS], tensor fil_b)
{
	double conv_val0, conv_val1, conv_val2, conv_val3;
	double alpha, bias;
	double beta0, beta1, beta2, beta3;
	double weight;

	double prev1, prev2, curr1, curr2;

	double input_pixel0;
	double input_pixel1;
	double input_pixel2;
	double input_pixel3;

	for (int b = 0; b < batch_size; ++b)
	{

		for (int f = 0; f < NUM_FILS; ++f)
		{

			alpha = alphas[f];
			bias = fil_b.data[f];

			for (int r = 0; r+1 < N_ROWS_CONV; r=r+2)
			{

				for (int c = 0; c+1 < N_COLS_CONV; c=c+2)
				{

					beta0 = betas[b][r  ][c  ];
					beta1 = betas[b][r  ][c+1];
					beta2 = betas[b][r+1][c  ];
					beta3 = betas[b][r+1][c+1];

					conv_val0 = 0.0;
					conv_val1 = 0.0;
					conv_val2 = 0.0;
					conv_val3 = 0.0;

					for (int i = 0; i < FIL_ROWS; ++i)
					{

						prev1 = bin_input_images[b][i+r  ][0+c  ];
						prev2 = bin_input_images[b][i+r+1][0+c  ];

						for (int j = 0; j < FIL_COLS; ++j)
						{

							INCREMENT_FLOPS(8)

							weight = fil_bin_w[f][i][j];

							input_pixel0 = prev1;
							input_pixel1 = bin_input_images[b][i+r  ][j+c+1];
							input_pixel2 = prev2;
							input_pixel3 = bin_input_images[b][i+r+1][j+c+1];

							// XNOR operation
							//conv_val += ( bin_input_images[b][i+r][j+c] == fil_bin_w[f][i][j] );

							conv_val0 += ( input_pixel0 * weight );
							conv_val1 += ( input_pixel1 * weight );
							conv_val2 += ( input_pixel2 * weight );
							conv_val3 += ( input_pixel3 * weight );

							prev1 = input_pixel1;
							prev2 = input_pixel3;
						}
					}

					INCREMENT_FLOPS(16)

					conv_val0 *= alpha * beta0;
					conv_val1 *= alpha * beta1;
					conv_val2 *= alpha * beta2;
					conv_val3 *= alpha * beta3;

					conv_val0 += bias;
					conv_val1 += bias;
					conv_val2 += bias;
					conv_val3 += bias;

					// applying ReLU
					if (conv_val0 < 0.0)
					{
						conv_val0 = 0.0;
					}

					if (conv_val1 < 0.0)
					{
						conv_val1 = 0.0;
					}

					if (conv_val2 < 0.0)
					{
						conv_val2 = 0.0;
					}

					if (conv_val3 < 0.0)
					{
						conv_val3 = 0.0;
					}

					(conv_t->data)[ind_conv_out(b, f, r  , c  )] = conv_val0;
					(conv_t->data)[ind_conv_out(b, f, r  , c+1)] = conv_val1;
					(conv_t->data)[ind_conv_out(b, f, r+1, c  )] = conv_val2;
					(conv_t->data)[ind_conv_out(b, f, r+1, c+1)] = conv_val3;
				}
			}
			
		}
	}
}*/




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