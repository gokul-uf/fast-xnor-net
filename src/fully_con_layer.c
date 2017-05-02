#include "fully_con_layer.h"

int N_ROWS_POOL;
int N_COLS_POOL;

void feed_forward(tensor* pool_t, tensor* fully_con_out, tensor* fully_con_w, tensor* fully_con_b){

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		for (int d = 0; d < N_DIGS; ++d)
		{
			double sum = (fully_con_b->data)[offset(fully_con_b, 0,0,0,d)];

			for (int f = 0; f < NUM_FILS; ++f)
			{
				for (int i = 0; i < N_ROWS_POOL; ++i)
				{
					for (int j = 0; j < N_COLS_POOL; ++j)
					{
						sum += (pool_t->data)[offset(pool_t, b, j, i, f)] * (fully_con_w->data)[offset(fully_con_w, d, j, i, f)];
					}
				
				}
			}
	
			(fully_con_out->data)[offset(fully_con_out, 0, 0, d, b)] = sum;
		}
	}
}

void initialize_weights_biases(tensor* fully_con_w, tensor* fully_con_b){
	for (int d = 0; d < N_DIGS; ++d)
	{
		(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, d)] = 1.0;

		for (int k = 0; k < NUM_FILS; ++k)
		{
			for (int i = 0; i < N_ROWS_POOL; ++i)
			{
				for (int j = 0; j < N_COLS_POOL; ++j)
				{
					(fully_con_w->data)[offset(fully_con_w, d, j, i, k)] = 1.0;
				}			
			}
		}
	}
}