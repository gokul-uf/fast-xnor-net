#include "xnornet.h"

void binarize_filters(tensor* fil_w, int bin_fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[]){
	int n = FIL_ROWS * FIL_COLS;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		double sum = 0.0, abs_val;
		for (int r = 0; r < FIL_ROWS; ++r)
		{
			for (int c = 0; c < FIL_COLS; ++c)
			{
				INCREMENT_FLOPS(4)
				double mat_val = (fil_w->data)[offset(fil_w, f, c, r, 0)];

				if (mat_val < 0.0)
				{
					bin_fil_w[f][r][c] = -1;
				}
				else{
					bin_fil_w[f][r][c] = 1;
				}

				abs_val = mat_val;
				if (abs_val < 0.0)
				{
					abs_val = -1.0 * abs_val;
				}

				sum += abs_val;
			}
		}
		INCREMENT_FLOPS(1)
		alphas[f] = sum/n;
	}
}

void bin_activation(tensor* input_images, int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], int shuffle_index[],
					double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV], int batch_size, int base){
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
						INCREMENT_FLOPS(3)
						double mat_val = (input_images->data)[offset(input_images, shuffle_index[b+base], j+c, i+r, 0)];

						if (mat_val < 0.0)
						{
							mat_val = -1.0 * mat_val;
						}

						sum += mat_val;
					}
				}
				INCREMENT_FLOPS(1)
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
				INCREMENT_FLOPS(1)
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
