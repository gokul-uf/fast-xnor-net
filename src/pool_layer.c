#include "pool_layer.h"

int N_ROWS_POOL;
int N_COLS_POOL;
int TOTAL_FLOPS;


// no loop unrolling
void max_pooling(tensor* conv_t, tensor* pool_t, int pool_index_i[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], 
	int pool_index_j[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int batch_size, char mode){

	int pool_i, pool_j;
	double max;
	int max_i, max_j;
	double conv_val_0, conv_val_1, conv_val_2, conv_val_3;

	for (int b = 0; b < batch_size; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int i = 0, pool_i = 0; i < N_ROWS_CONV; i=i+2, ++pool_i)
			{

				for (int j = 0, pool_j = 0; j < N_COLS_CONV; j=j+2, ++pool_j)
				{
					INCREMENT_FLOPS(4)

					conv_val_0 = (conv_t->data)[ind_conv_out(b, f, i  , j  )];
					conv_val_1 = (conv_t->data)[ind_conv_out(b, f, i  , j+1)];
					conv_val_2 = (conv_t->data)[ind_conv_out(b, f, i+1, j  )];
					conv_val_3 = (conv_t->data)[ind_conv_out(b, f, i+1, j+1)];

					max = 0.0;
					max_i = i, max_j = j;

					if (conv_val_0 > max)
					{
						max   = conv_val_0;
						max_i = i;
						max_j = j;
					}

					if (conv_val_1 > max)
					{
						max   = conv_val_1;
						max_i = i;
						max_j = j+1;
					}

					if (conv_val_2 > max)
					{
						max   = conv_val_2;
						max_i = i+1;
						max_j = j;
					}

					if (conv_val_3 > max)
					{
						max   = conv_val_3;
						max_i = i+1;
						max_j = j+1;
					}

					(pool_t->data)[ind_pool_out(b, f, pool_i, pool_j)] = max;

					// Only if mode is training
					if (mode == 'T')
					{
						pool_index_i[b][f][pool_i][pool_j] = max_i;
						pool_index_j[b][f][pool_i][pool_j] = max_j;
					}
				}
			}
		}
	}
}
