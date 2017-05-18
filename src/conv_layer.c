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

// No loop unrolling
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
}

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
/*void bin_convolution(tensor* input_t, tensor* conv_t, tensor* pool_t, int batch_size,
	int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[],
	int pool_index_i[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double conv_val_r0c0f0, conv_val_r0c0f1, conv_val_r0c0f2;
	double conv_val_r0c1f0, conv_val_r0c1f1, conv_val_r0c1f2;
	double conv_val_r1c0f0, conv_val_r1c0f1, conv_val_r1c0f2;
	double conv_val_r1c1f0, conv_val_r1c1f1, conv_val_r1c1f2;

	double input_pixel0, input_pixel1, input_pixel2, input_pixel3;

	int cur_image;

	double alpha_0 = alphas[0];
	double alpha_1 = alphas[1];
	double alpha_2 = alphas[2];

	double bias_0 = fil_b.data[offset(&fil_b, 0, 0, 0, 0)];
	double bias_1 = fil_b.data[offset(&fil_b, 1, 0, 0, 0)];
	double bias_2 = fil_b.data[offset(&fil_b, 2, 0, 0, 0)];

	double max0_1, max0_2, max0;
	int max0_1_i, max0_1_j, max0_2_i, max0_2_j, max0_i, max0_j;

	double max1_1, max1_2, max1;
	int max1_1_i, max1_1_j, max1_2_i, max1_2_j, max1_i, max1_j;

	double max2_1, max2_2, max2;
	int max2_1_i, max2_1_j, max2_2_i, max2_2_j, max2_i, max2_j;

	int pool_i, pool_j;


	for (int b = 0; b < batch_size; ++b)
	{
		cur_image = shuffle_index[b+base];

		// Unroll r and c loops by 2 so that max pooling can be merged with convolution
		for (int r = 0, pool_i=0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
		{
			for (int c = 0, pool_j=0; c+1 < N_ROWS_CONV; c=c+2, ++pool_j)
			{
				conv_val_r0c0f0 = 0.0, conv_val_r0c0f1 = 0.0, conv_val_r0c0f2 = 0.0;
				conv_val_r0c1f0 = 0.0, conv_val_r0c1f1 = 0.0, conv_val_r0c1f2 = 0.0;
				conv_val_r1c0f0 = 0.0, conv_val_r1c0f1 = 0.0, conv_val_r1c0f2 = 0.0;
				conv_val_r1c1f0 = 0.0, conv_val_r1c1f1 = 0.0, conv_val_r1c1f2 = 0.0;

				for (int i = 0; i < FIL_ROWS; ++i)
				{

					for (int j = 0; j+1 < FIL_COLS; ++j)
					{
						input_pixel0 = (input_t->data)[offset(input_t, cur_image, c+j  , r+i  , 0)];
						input_pixel1 = (input_t->data)[offset(input_t, cur_image, c+j+1, r+i  , 0)];
						input_pixel2 = (input_t->data)[offset(input_t, cur_image, c+j  , r+i+1, 0)];
						input_pixel3 = (input_t->data)[offset(input_t, cur_image, c+j+1, r+i+1, 0)];

						INCREMENT_FLOPS(12)
						// --------------------------------------------filter 0-------------------------------------
						if (fil_bin_w[0][i][j] == 1)
						{
							conv_val_r0c0f0 += input_pixel0;
							conv_val_r0c1f0 += input_pixel1;
							conv_val_r1c0f0 += input_pixel2;
							conv_val_r1c0f0 += input_pixel3;
						}
						else
						{
							conv_val_r0c0f0 -= input_pixel0;
							conv_val_r0c1f0 -= input_pixel1;
							conv_val_r1c0f0 -= input_pixel2;
							conv_val_r1c0f0 -= input_pixel3;
						}

						// --------------------------------------------filter 1-----------------------------------
						if (fil_bin_w[1][i][j] == 1)
						{
							conv_val_r0c0f1 += input_pixel0;
							conv_val_r0c1f1 += input_pixel1;
							conv_val_r1c0f1 += input_pixel2;
							conv_val_r1c0f1 += input_pixel3;
						}
						else
						{
							conv_val_r0c0f1 -= input_pixel0;
							conv_val_r0c1f1 -= input_pixel1;
							conv_val_r1c0f1 -= input_pixel2;
							conv_val_r1c0f1 -= input_pixel3;
						}

						// -------------------------------------------filter 2----------------------------------------------
						if (fil_bin_w[2][i][j] == 1)
						{
							conv_val_r0c0f2 += input_pixel0;
							conv_val_r0c1f2 += input_pixel1;
							conv_val_r1c0f2 += input_pixel2;
							conv_val_r1c0f2 += input_pixel3;
						}
						else
						{
							conv_val_r0c0f2 -= input_pixel0;
							conv_val_r0c1f2 -= input_pixel1;
							conv_val_r1c0f2 -= input_pixel2;
							conv_val_r1c0f2 -= input_pixel3;
						}

					}
				}
			
				INCREMENT_FLOPS(48)

				// -----------------------------------------------filter 0 ----------------------------------------------
				conv_val_r0c0f0 *= alpha_0;
				conv_val_r0c0f0 += bias_0;

				conv_val_r0c1f0 *= alpha_0;
				conv_val_r0c1f0 += bias_0;

				conv_val_r1c0f0 *= alpha_0;
				conv_val_r1c0f0 += bias_0;

				conv_val_r1c1f0 *= alpha_0;
				conv_val_r1c1f0 += bias_0;

				// -----------------------------------------------filter 1---------------------------------------------
				conv_val_r0c0f1 *= alpha_0;
				conv_val_r0c0f1 += bias_0;

				conv_val_r0c1f1 *= alpha_0;
				conv_val_r0c1f1 += bias_0;

				conv_val_r1c0f1 *= alpha_0;
				conv_val_r1c0f1 += bias_0;

				conv_val_r1c1f1 *= alpha_0;
				conv_val_r1c1f1 += bias_0;

				// -----------------------------------------------filter 2---------------------------------------------
				conv_val_r0c0f1 *= alpha_0;
				conv_val_r0c0f1 += bias_0;

				conv_val_r0c1f1 *= alpha_0;
				conv_val_r0c1f1 += bias_0;

				conv_val_r1c0f1 *= alpha_0;
				conv_val_r1c0f1 += bias_0;

				conv_val_r1c1f1 *= alpha_0;
				conv_val_r1c1f1 += bias_0;

				// applying ReLU
				// -------------------------------------------filter 0------------------------------------------------
				if (conv_val_r0c0f0 < 0.0)
				{
					conv_val_r0c0f0 = 0.0;
				}

				if (conv_val_r0c1f0 < 0.0)
				{
					conv_val_r0c1f0 = 0.0;
				}

				if (conv_val_r1c0f0 < 0.0)
				{
					conv_val_r1c0f0 = 0.0;
				}

				if (conv_val_r1c1f0 < 0.0)
				{
					conv_val_r1c1f0 = 0.0;
				}

				// -------------------------------------------filter 1------------------------------------------------
				if (conv_val_r0c0f1 < 0.0)
				{
					conv_val_r0c0f1 = 0.0;
				}

				if (conv_val_r0c1f1 < 0.0)
				{
					conv_val_r0c1f1 = 0.0;
				}

				if (conv_val_r1c0f1 < 0.0)
				{
					conv_val_r1c0f1 = 0.0;
				}

				if (conv_val_r1c1f1 < 0.0)
				{
					conv_val_r1c1f1 = 0.0;
				}

				// -------------------------------------------filter 2------------------------------------------------
				if (conv_val_r0c0f1 < 0.0)
				{
					conv_val_r0c0f1 = 0.0;
				}

				if (conv_val_r0c1f1 < 0.0)
				{
					conv_val_r0c1f1 = 0.0;
				}

				if (conv_val_r1c0f1 < 0.0)
				{
					conv_val_r1c0f1 = 0.0;
				}

				if (conv_val_r1c1f1 < 0.0)
				{
					conv_val_r1c1f1 = 0.0;
				}
			
				(conv_t->data)[offset(conv_t,b,c  ,r  ,0)] = conv_val_r0c0f0;
				(conv_t->data)[offset(conv_t,b,c+1,r  ,0)] = conv_val_r0c1f0;
				(conv_t->data)[offset(conv_t,b,c  ,r+1,0)] = conv_val_r1c0f0;
				(conv_t->data)[offset(conv_t,b,c+1,r+1,0)] = conv_val_r1c1f0;

				(conv_t->data)[offset(conv_t,b,c  ,r  ,1)] = conv_val_r0c0f1;
				(conv_t->data)[offset(conv_t,b,c+1,r  ,1)] = conv_val_r0c1f1;
				(conv_t->data)[offset(conv_t,b,c  ,r+1,1)] = conv_val_r1c0f1;
				(conv_t->data)[offset(conv_t,b,c+1,r+1,1)] = conv_val_r1c1f1;

				(conv_t->data)[offset(conv_t,b,c  ,r  ,2)] = conv_val_r0c0f2;
				(conv_t->data)[offset(conv_t,b,c+1,r  ,2)] = conv_val_r0c1f2;
				(conv_t->data)[offset(conv_t,b,c  ,r+1,2)] = conv_val_r1c0f2;
				(conv_t->data)[offset(conv_t,b,c+1,r+1,2)] = conv_val_r1c1f2;

				// --------------------------------------------Max Pooling-------------------------------------
				INCREMENT_FLOPS(9)

				
				// -------------------------------------------Filter 0----------------------------------------
				if (conv_val_r0c0f0 > conv_val_r0c1f0)
				{
					max0_1 = conv_val_r0c0f0;
					max0_1_i = r;
					max0_1_j = c;
				}
				else
				{
					max0_1 = conv_val_r0c1f0;
					max0_1_i = r;
					max0_1_j = c+1;
				}

				if (conv_val_r1c0f0 > conv_val_r1c1f0)
				{
					max0_2 = conv_val_r1c0f0;
					max0_2_i = r+1;
					max0_2_j = c;
				}
				else
				{
					max0_2 = conv_val_r1c1f0;
					max0_2_i = r+1;
					max0_2_j = c+1;
				}

				if (max0_1 > max0_2)
				{
					max0 = max0_1;
					max0_i = max1_i;
					max0_j = max1_j;
				}
				else
				{
					max0 = max0_2;
					max0_i = max0_2_i;
					max0_j = max0_2_j;
				}

				(pool_t->data)[offset(pool_t,b,pool_j,pool_i,0)] = max0;
				pool_index_i[b][0][pool_i][pool_j] = max0_i;
				pool_index_j[b][0][pool_i][pool_j] = max0_j;


				// -------------------------------------------Filter 1----------------------------------------
				if (conv_val_r0c0f1 > conv_val_r0c1f1)
				{
					max1_1 = conv_val_r0c0f1;
					max1_1_i = r;
					max1_1_j = c;
				}
				else
				{
					max1_1 = conv_val_r0c1f1;
					max1_1_i = r;
					max1_1_j = c+1;
				}

				if (conv_val_r1c0f1 > conv_val_r1c1f1)
				{
					max1_2 = conv_val_r1c0f1;
					max1_2_i = r+1;
					max1_2_j = c;
				}
				else
				{
					max1_2 = conv_val_r1c1f1;
					max1_2_i = r+1;
					max1_2_j = c+1;
				}

				if (max1_1 > max1_2)
				{
					max1   = max1_1;
					max1_i = max1_1_i;
					max1_j = max1_1_j;
				}
				else
				{
					max1   = max1_2;
					max1_i = max1_2_i;
					max1_j = max1_2_j;
				}

				(pool_t->data)[offset(pool_t,b,pool_j,pool_i,1)] = max1;
				pool_index_i[b][1][pool_i][pool_j] = max1_i;
				pool_index_j[b][1][pool_i][pool_j] = max1_j;

				// -------------------------------------------Filter 2----------------------------------------
				if (conv_val_r0c0f2 > conv_val_r0c1f2)
				{
					max2_1 = conv_val_r0c0f2;
					max2_1_i = r;
					max2_1_j = c;
				}
				else
				{
					max2_1 = conv_val_r0c1f2;
					max2_1_i = r;
					max2_1_j = c+1;
				}

				if (conv_val_r1c0f2 > conv_val_r1c1f2)
				{
					max2_2 = conv_val_r1c0f2;
					max2_2_i = r+1;
					max2_2_j = c;
				}
				else
				{
					max2_2 = conv_val_r1c1f2;
					max2_2_i = r+1;
					max2_2_j = c+1;
				}

				if (max2_1 > max2_2)
				{
					max2   = max2_1;
					max2_i = max2_1_i;
					max2_j = max2_1_j;
				}
				else
				{
					max2   = max2_2;
					max2_i = max2_2_i;
					max2_j = max2_2_j;
				}

				(pool_t->data)[offset(pool_t,b,pool_j,pool_i,2)] = max2;
				pool_index_i[b][2][pool_i][pool_j] = max2_i;
				pool_index_j[b][2][pool_i][pool_j] = max2_j;
			}
		}
	}
}*/

// loop on conv rows and cols unrolled by 2, max-pooling done, pool indexes saved
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
void xnor_convolution(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], 
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
}

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