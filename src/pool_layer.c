#include "pool_layer.h"

int N_ROWS_POOL;
int N_COLS_POOL;

void max_pooling(tensor* conv_t, tensor* pool_t, int pool_index_i[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL],
	int pool_index_j[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int batch_size, char mode){

	int pool_i, pool_j;
	for (int b = 0; b < batch_size; ++b)
	{
		COST_INC_I_ADD(1);
		for (int f = 0; f < NUM_FILS; ++f)
		{
			COST_INC_I_ADD(1);
			for (int i = 0, pool_i = 0; i < N_ROWS_CONV; i=i+2, ++pool_i)
			{
				COST_INC_I_ADD(1);
				double max = 0.0;
				int max_i = 0, max_j = 0;

				for (int j = 0, pool_j = 0; j < N_COLS_CONV; j=j+2, ++pool_j)
				{
					COST_INC_I_ADD(1);
					double max = 0.0;
					int max_i = i, max_j = j;

					if ((conv_t->data)[offset(conv_t,b,j,i,f)] > max)
					{
						max = (conv_t->data)[offset(conv_t,b,j,i,f)];
						max_i = i;
						max_j = j;
					}
					COST_INC_I_ADD(1);
					if ((conv_t->data)[offset(conv_t,b,j+1,i,f)] > max)
					{
						COST_INC_I_ADD(2);
						max = (conv_t->data)[offset(conv_t,b,j+1,i,f)];
						max_i = i;
						max_j = j+1;
					}
					COST_INC_I_ADD(1);
					if ((conv_t->data)[offset(conv_t,b,j,i+1,f)] > max)
					{
						COST_INC_I_ADD(2);
						max = (conv_t->data)[offset(conv_t,b,j,i+1,f)];
						max_i = i+1;
						max_j = j;
					}
					COST_INC_I_ADD(2);
					if ((conv_t->data)[offset(conv_t,b,j+1,i+1,f)] > max)
					{
						COST_INC_I_ADD(4);
						max = (conv_t->data)[offset(conv_t,b,j+1,i+1,f)];
						max_i = i+1;
						max_j = j+1;
					}

					(pool_t->data)[offset(pool_t,b,pool_j,pool_i,f)] = max;

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
