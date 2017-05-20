#include "xnornet.h"

int TOTAL_FLOPS;

// loop on filter rows and cols unrolled by 2
void binarize_filters(tensor* fil_w, int bin_fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[])
{
	int n = FIL_ROWS * FIL_COLS;
	double mat_val_1;
	double mat_val_2;
	double mat_val_3;
	double mat_val_4;
	double mat_val;

	double sum1, sum2, sum3, sum4, sum_rem, sum; 
	double abs_val1, abs_val2, abs_val3, abs_val4, abs_val_rem;

	int r, c;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, sum_rem = 0.0, sum = 0.0; 
		
		for (r = 0; r+1 < FIL_ROWS; r=r+2)
		{
			for (c = 0; c+1 < FIL_COLS; c=c+2)
			{
				INCREMENT_FLOPS(12)

				mat_val_1 = (fil_w->data)[ind_fil_w(f, r  , c  )];
				mat_val_2 = (fil_w->data)[ind_fil_w(f, r  , c+1)];
				mat_val_3 = (fil_w->data)[ind_fil_w(f, r+1, c  )];
				mat_val_4 = (fil_w->data)[ind_fil_w(f, r+1, c+1)];

				if (mat_val_1 < 0.0)
				{
					abs_val1 = -mat_val_1;
					bin_fil_w[f][r][c] = -1;
				}
				else{
					abs_val1 = mat_val_1;
					bin_fil_w[f][r][c] = 1;
				}

				if (mat_val_2 < 0.0)
				{
					abs_val2 = -mat_val_2;
					bin_fil_w[f][r][c+1] = -1;
				}
				else{
					abs_val2 = mat_val_2;
					bin_fil_w[f][r][c+1] = 1;
				}

				if (mat_val_3 < 0.0)
				{
					abs_val3 = -mat_val_3;
					bin_fil_w[f][r+1][c] = -1;
				}
				else{
					abs_val3 = mat_val_3;
					bin_fil_w[f][r+1][c] = 1;
				}

				if (mat_val_4 < 0.0)
				{
					abs_val4 = -mat_val_4;
					bin_fil_w[f][r+1][c+1] = -1;
				}
				else{
					abs_val4 = mat_val_4;
					bin_fil_w[f][r+1][c+1] = 1;
				}

				sum1 += abs_val1;
				sum2 += abs_val2;
				sum3 += abs_val3;
				sum4 += abs_val4;
			}

			// left over in the current row, at the end
			for (; c < FIL_COLS; ++c)
			{
				INCREMENT_FLOPS(3)

				mat_val = (fil_w->data)[ind_fil_w(f, r, c)];

				if (mat_val < 0.0)
				{
					abs_val_rem = -mat_val;
					bin_fil_w[f][r][c] = -1;
				}
				else{
					abs_val_rem = mat_val;
					bin_fil_w[f][r][c] = 1;
				}

				sum_rem += abs_val_rem;
			}
		}

		INCREMENT_FLOPS(4)
		sum = sum1 + sum2 + sum3 + sum4 + sum_rem;

		// left over rows, in the end
		for (; r < FIL_ROWS; ++r)
		{
			for (c = 0; c < FIL_COLS; ++c)
			{
				INCREMENT_FLOPS(3)

				mat_val = (fil_w->data)[ind_fil_w(f, r, c)];

				if (mat_val < 0.0)
				{
					abs_val_rem = -mat_val;
					bin_fil_w[f][r][c] = -1;
				}
				else{
					abs_val_rem = mat_val;
					bin_fil_w[f][r][c] = 1;
				}

				sum += abs_val_rem;
			}
		}

		INCREMENT_FLOPS(1)
		alphas[f] = sum/n;
	}
}

// using previous loads to reduce loading more data
void bin_activation(tensor* input_images, int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], int shuffle_index[], 
					double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], int batch_size, int base)
{
	int n = FIL_ROWS * FIL_COLS;
	int cur_image;
	double sum1, sum2, sum3, sum4;
	double mat_val;
	double abs_input[batch_size][IMAGE_ROWS][IMAGE_COLS];
	double abs_val;

	double prev1, prev2, curr1, curr2;

	for (int b = 0; b < batch_size; ++b)
	{
		cur_image = shuffle_index[b+base];

		for (int i = 0; i < IMAGE_ROWS; ++i)
		{
			for (int j = 0; j < IMAGE_COLS; ++j)
			{
				INCREMENT_FLOPS(2)

				mat_val = (input_images->data)[ind_input_img(cur_image, i, j)];

				if (mat_val < 0.0)
				{
					abs_val = -mat_val;
					bin_input_images[b][i][j] = -1;
				}
				else
				{
					abs_val = mat_val;
					bin_input_images[b][i][j] = 1;
				}

				abs_input[b][i][j] = abs_val;
			}
		}
	}

	for (int b = 0; b < batch_size; ++b)
	{
		cur_image = shuffle_index[b+base];

		for (int i = 0; i+1 < N_ROWS_CONV; i=i+2)
		{

			for (int j = 0; j+1 < N_COLS_CONV; j=j+2)
			{

				sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
				for (int r = 0; r < FIL_ROWS; ++r)
				{

					prev1 = abs_input[b][i  +r][j  +0];
					prev2 = abs_input[b][i+1+r][j  +0];

					for (int c = 0; c < FIL_COLS; ++c)
					{
						INCREMENT_FLOPS(4)

						curr1 = abs_input[b][i  +r][j+1+c];
						curr2 = abs_input[b][i+1+r][j+1+c];

						sum1 += prev1;
						sum2 += curr1;
						sum3 += prev2;
						sum4 += curr2;

						prev1 = curr1;
						prev2 = curr2;
					}

				}

				INCREMENT_FLOPS(4)

				betas[b][i  ][j  ] = sum1/n;
				betas[b][i  ][j+1] = sum2/n;
				betas[b][i+1][j  ] = sum3/n;
				betas[b][i+1][j+1] = sum4/n;
			}
		}
	}
}
