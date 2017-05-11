#include "xnornet.h"

void binarize_filters(tensor* fil_w, int bin_fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[]){
	int n = FIL_ROWS * FIL_COLS;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, sum_rem = 0.0, sum; 
		double abs_val1, abs_val2, abs_val3, abs_val4, abs_val_rem;

		int r, c;
		
		for (r = 0; r+1 < FIL_ROWS; r=r+2)
		{
			for (c = 0; c+1 < FIL_COLS; c=c+2)
			{
				double mat_val_1 = (fil_w->data)[offset(fil_w, f, c  , r  , 0)];
				double mat_val_2 = (fil_w->data)[offset(fil_w, f, c+1, r  , 0)];
				double mat_val_3 = (fil_w->data)[offset(fil_w, f, c  , r+1, 0)];
				double mat_val_4 = (fil_w->data)[offset(fil_w, f, c+1, r+1, 0)];

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
				double mat_val = (fil_w->data)[offset(fil_w, f, c, r, 0)];

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

		sum = sum1 + sum2 + sum3 + sum4 + sum_rem;

		// left over rows, in the end
		for (; r < FIL_ROWS; ++r)
		{
			for (c = 0; c < FIL_COLS; ++c)
			{
				double mat_val = (fil_w->data)[offset(fil_w, f, c, r, 0)];

				if (mat_val < 0.0)
				{
					abs_val1 = -mat_val;
					bin_fil_w[f][r][c] = -1;
				}
				else{
					abs_val1 = mat_val;
					bin_fil_w[f][r][c] = 1;
				}

				sum += abs_val1;
			}
		}

		alphas[f] = sum/n;
	}
}

void bin_activation(tensor* input_images, int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], int shuffle_index[], 
					double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], int batch_size, int base)
{
	int n = FIL_ROWS * FIL_COLS;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		for (int i = 0; i < N_ROWS_CONV; ++i)
		{
			for (int j = 0; j < N_COLS_CONV; ++j)
			{
				double sum = 0.0;
				for (int r = 0; r < FIL_ROWS; ++r)
				{
					for (int c = 0; c < FIL_COLS; ++c)
					{
						double mat_val = (input_images->data)[offset(input_images, shuffle_index[b+base], j+c, i+r, 0)];

						if (mat_val < 0.0)
						{
							mat_val = -1.0 * mat_val;
						}

						sum += mat_val;
					}
				}

				betas[b][i][j] = sum/n;
			}
		}
	}

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		for (int r = 0; r < IMAGE_ROWS; ++r)
		{
			for (int c = 0; c < IMAGE_COLS; ++c)
			{
				double mat_val = (input_images->data)[offset(input_images, shuffle_index[b+base], c, r, 0)];

				if (mat_val < 0.0)
				{
					bin_input_images[b][r][c] = -1;
				}
				else{
					bin_input_images[b][r][c] = 1;
				}
			}
		}
	}
}