#include "xnornet.h"

void binarize(double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], int bin_fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[]){
	int n = FIL_ROWS * FIL_COLS;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		double sum = 0.0, abs_val;
		for (int r = 0; r < FIL_ROWS; ++r)
		{
			for (int c = 0; c < FIL_COLS; ++c)
			{

				if (fil_w[f][r][c] < 0)
				{
					bin_fil_w[f][r][c] = -1;
				}
				else{
					bin_fil_w[f][r][c] = 1;
				}

				abs_val = fil_w[f][r][c];
				if (abs_val < 0.0)
				{
					abs_val = -1.0 * abs_val;
				}

				sum += abs_val;
			}
		}

		alphas[f] = sum/n;
	}
}