#include "fully_con_layer.h"

int N_ROWS_POOL;
int N_COLS_POOL;

void feed_forward(tensor* pool_t, tensor* fully_con_out, tensor* fully_con_w, tensor* fully_con_b, int batch_size){

	for (int b = 0; b < batch_size; ++b)
	{
		COST_INC_I_ADD(1);
		for (int d = 0; d < N_DIGS; ++d)
		{
			COST_INC_I_ADD(1);
			double sum = (fully_con_b->data)[offset(fully_con_b, 0,0,0,d)];//Initialize with bias

			for (int f = 0; f < NUM_FILS; ++f)
			{
				COST_INC_I_ADD(1);
				for (int i = 0; i < N_ROWS_POOL; ++i)
				{
					COST_INC_I_ADD(1);
					for (int j = 0; j < N_COLS_POOL; ++j)
					{
						COST_INC_I_ADD(1);

						COST_INC_F_ADD(1); COST_INC_I_MUL(1);
						sum += (pool_t->data)[offset(pool_t, b, j, i, f)] * (fully_con_w->data)[offset(fully_con_w, d, j, i, f)];
					}

				}
			}

			(fully_con_out->data)[offset(fully_con_out, b, 0, 0, d)] = sum;
		}
	}
}

void softmax(tensor* fully_con_out, tensor* softmax_out, int preds[], int batch_size){
	for (int b = 0; b < batch_size; ++b)
	{
		COST_INC_I_ADD(1);
		double max = -99999.0, max_index = -1;
		for (int d = 0; d < N_DIGS; ++d)
		{
			COST_INC_I_ADD(1);
			if ((fully_con_out->data)[offset(fully_con_out, b, 0, 0, d)] > max)
			{
				max = (fully_con_out->data)[offset(fully_con_out, b, 0, 0, d)];
				max_index = d;
			}
		}

		preds[b] = max_index;

		double exp_sum = 0.0;
		for (int d = 0; d < N_DIGS; ++d)
		{
			COST_INC_I_ADD(1);
			COST_INC_F_ADD(2);
			exp_sum += exp( (fully_con_out->data)[offset(fully_con_out, b, 0, 0, d)] - max );
		}

		for (int d = 0; d < N_DIGS; ++d)
		{
			COST_INC_I_ADD(1);
			COST_INC_F_ADD(2);
			(softmax_out->data)[offset(softmax_out, b, 0, 0, d)] = exp(
																	(fully_con_out->data)[offset(fully_con_out, b, 0, 0, d)] - max
																	- log( exp_sum )
																	);
		}
	}
}

void initialize_weights_biases(tensor* fully_con_w, tensor* fully_con_b){

	srand( time(NULL) );

	for (int d = 0; d < N_DIGS; ++d)
	{
		COST_INC_I_ADD(1);
		(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, d)] = 0.0;

		for (int k = 0; k < NUM_FILS; ++k)
		{
			COST_INC_I_ADD(1);
			for (int i = 0; i < N_ROWS_POOL; ++i)
			{
				COST_INC_I_ADD(1);
				for (int j = 0; j < N_COLS_POOL; ++j)
				{
					COST_INC_I_ADD(1);
					int r = rand();
					COST_INC_I_OTHER(1);
					if (r%2 == 0)
					{
						COST_INC_F_DIV(1);
						double ran = ((double)rand())/RAND_MAX;
						(fully_con_w->data)[offset(fully_con_w, d, j, i, k)] = -ran;
					}
					else
					{
						COST_INC_F_DIV(1);
						double ran = ((double)rand())/RAND_MAX;
						(fully_con_w->data)[offset(fully_con_w, d, j, i, k)] = ran;
					}
				}
			}
		}
	}
}
