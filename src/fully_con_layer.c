#include "fully_con_layer.h"

int N_ROWS_POOL;
int N_COLS_POOL;
int TOTAL_FLOPS;

// no loop unrolling
void feed_forward(tensor* pool_t, tensor* fully_con_out, tensor* fully_con_w, tensor* fully_con_b, int batch_size)
{
	double sum;

	for (int b = 0; b < batch_size; ++b)
	{
		for (int d = 0; d < N_DIGS; ++d)
		{
			sum = (fully_con_b->data)[d];//Initialize with bias

			for (int f = 0; f < NUM_FILS; ++f)
			{
				for (int i = 0; i < N_ROWS_POOL; ++i)
				{
					for (int j = 0; j < N_COLS_POOL; ++j)
					{
						INCREMENT_FLOPS(2)

						sum += (pool_t->data)[ind_pool_out(b, f, i, j)] * (fully_con_w->data)[ind_fully_con_w(d, f, i, j)];
					}
				
				}
			}
	
			(fully_con_out->data)[ind_fully_con_out(b, d)] = sum;
		}
	}
}

// no loop unrolling
void softmax(tensor* fully_con_out, tensor* softmax_out, int preds[], int batch_size)
{
	double max, max_index;
	double mat_val;
	double exp_sum;

	for (int b = 0; b < batch_size; ++b)
	{
		max = -99999.0, max_index = -1;
		for (int d = 0; d < N_DIGS; ++d)
		{
			INCREMENT_FLOPS(1)

			mat_val = (fully_con_out->data)[ind_fully_con_out(b, d)];

			if (mat_val > max)
			{
				max = mat_val;
				max_index = d;
			}
		}

		preds[b] = max_index;

		exp_sum = 0.0;
		for (int d = 0; d < N_DIGS; ++d)
		{
			INCREMENT_FLOPS(3)

			exp_sum += exp( (fully_con_out->data)[ind_fully_con_out(b, d)] - max );
		}

		for (int d = 0; d < N_DIGS; ++d)
		{
			INCREMENT_FLOPS(4)

			(softmax_out->data)[ind_softmax_out(b, d)] = exp(
																(fully_con_out->data)[ind_fully_con_out(b, d)] - max
																- log( exp_sum )
															);
		}
	}
}

void initialize_weights_biases(tensor* fully_con_w, tensor* fully_con_b){

	srand( time(NULL) );
	int r;
	double ran;

	for (int d = 0; d < N_DIGS; ++d)
	{
		(fully_con_b->data)[d] = 0.0;

		for (int k = 0; k < NUM_FILS; ++k)
		{
			for (int i = 0; i < N_ROWS_POOL; ++i)
			{
				for (int j = 0; j < N_COLS_POOL; ++j)
				{
					r = rand();

					if (r%2 == 0)
					{
						ran = ((double)rand())/RAND_MAX;
						(fully_con_w->data)[ind_fully_con_w(d, k, i, j)] = -ran;
					}
					else
					{
						ran = ((double)rand())/RAND_MAX;
						(fully_con_w->data)[ind_fully_con_w(d, k, i, j)] = ran;
					}
				}			
			}
		}
	}
}



/*void feed_forward(tensor* pool_t, tensor* fully_con_out, tensor* fully_con_w, tensor* fully_con_b, int batch_size){

	for (int b = 0; b < batch_size; ++b)
	{
		for (int d = 0; d < N_DIGS; ++d)
		{
			double sum = (fully_con_b->data)[offset(fully_con_b, 0,0,0,d)];//Initialize with bias

			for (int f = 0; f < NUM_FILS; ++f)
			{
				for (int i = 0; i < N_ROWS_POOL; ++i)
				{
					for (int j = 0; j < N_COLS_POOL; ++j)
					{
						INCREMENT_FLOPS(2)

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
		double max = -99999.0, max_index = -1;
		for (int d = 0; d < N_DIGS; ++d)
		{
			INCREMENT_FLOPS(1)

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
			INCREMENT_FLOPS(3)

			exp_sum += exp( (fully_con_out->data)[offset(fully_con_out, b, 0, 0, d)] - max );
		}

		for (int d = 0; d < N_DIGS; ++d)
		{
			INCREMENT_FLOPS(4)

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
		(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, d)] = 0.0;

		for (int k = 0; k < NUM_FILS; ++k)
		{
			for (int i = 0; i < N_ROWS_POOL; ++i)
			{
				for (int j = 0; j < N_COLS_POOL; ++j)
				{
					int r = rand();

					if (r%2 == 0)
					{
						double ran = ((double)rand())/RAND_MAX;
						(fully_con_w->data)[offset(fully_con_w, d, j, i, k)] = -ran;
					}
					else
					{
						double ran = ((double)rand())/RAND_MAX;
						(fully_con_w->data)[offset(fully_con_w, d, j, i, k)] = ran;
					}
				}			
			}
		}
	}
}*/