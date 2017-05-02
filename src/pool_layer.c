#include "pool_layer.h"

void max_pooling(tensor* conv_t, tensor* pool_t, tensor* pool_index_t, int n_rows_conv, 
	int n_cols_conv, int batch_size, int base){
	for (int b = 0; b < batch_size; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int i = 0; i < n_rows_conv; i=i+2)
			{
				for (int j = 0; j < n_cols_conv; j=j+2)
				{
					//(conv_t->data)[offset(conv_t,b,j,i,f)] = convolve(input_t, i, j, b+base, fil_w, fil_b, f);
				}
			}
			
		}
	}
}
