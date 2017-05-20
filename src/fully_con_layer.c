#include "fully_con_layer.h"

int N_ROWS_POOL;
int N_COLS_POOL;
int TOTAL_FLOPS;

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

//// loops on pool rows and cols unrolled, loop on number of filters also unrolled
void feed_forward(tensor* pool_t, tensor* fully_con_out, tensor* fully_con_w, tensor* fully_con_b, int batch_size)
{

	double  sum_r0c0,  sum_r0c1,  sum_r1c0,  sum_r1c1;
	double pool_r0c0, pool_r0c1, pool_r1c0, pool_r1c1;
	double    w_r0c0,    w_r0c1,    w_r1c0,    w_r1c1;

	double  sum_r0c0_f0,  sum_r0c1_f0,  sum_r1c0_f0,  sum_r1c1_f0;
	double pool_r0c0_f0, pool_r0c1_f0, pool_r1c0_f0, pool_r1c1_f0;
	double    w_r0c0_f0,    w_r0c1_f0,    w_r1c0_f0,    w_r1c1_f0;

	double  sum_r0c0_f1,  sum_r0c1_f1,  sum_r1c0_f1,  sum_r1c1_f1;
	double pool_r0c0_f1, pool_r0c1_f1, pool_r1c0_f1, pool_r1c1_f1;
	double    w_r0c0_f1,    w_r0c1_f1,    w_r1c0_f1,    w_r1c1_f1;

	double  sum_r0c0_f2,  sum_r0c1_f2,  sum_r1c0_f2,  sum_r1c1_f2;
	double pool_r0c0_f2, pool_r0c1_f2, pool_r1c0_f2, pool_r1c1_f2;
	double    w_r0c0_f2,    w_r0c1_f2,    w_r1c0_f2,    w_r1c1_f2;

	double sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7, sum_8, sum_9, sum_10;
	double sum_f0, sum_f1, sum_f2, sum_rem, sum;

	int f;

	for (int b = 0; b < batch_size; ++b)
	{
		for (int d = 0; d < N_DIGS; ++d)
		{
			sum_r0c0 = 0.0, sum_r0c1 = 0.0, sum_r1c0 = 0.0, sum_r1c1 = 0.0;

			sum_r0c0_f0 = 0.0, sum_r0c1_f0 = 0.0, sum_r1c0_f0 = 0.0, sum_r1c1_f0 = 0.0;
			sum_r0c0_f1 = 0.0, sum_r0c1_f1 = 0.0, sum_r1c0_f1 = 0.0, sum_r1c1_f1 = 0.0;
			sum_r0c0_f2 = 0.0, sum_r0c1_f2 = 0.0, sum_r1c0_f2 = 0.0, sum_r1c1_f2 = 0.0;

			for (f = 0; f+2 < NUM_FILS; f=f+3)
			{
				for (int i = 0; i+1 < N_ROWS_POOL; i=i+2)
				{
					for (int j = 0; j+1 < N_COLS_POOL; j=j+2)
					{
						INCREMENT_FLOPS(24)
						pool_r0c0_f0 = (pool_t->data)[ind_pool_out(b, f  , i  , j  )]; 
						pool_r0c1_f0 = (pool_t->data)[ind_pool_out(b, f  , i  , j+1)];
						pool_r1c0_f0 = (pool_t->data)[ind_pool_out(b, f  , i+1, j  )];
						pool_r1c1_f0 = (pool_t->data)[ind_pool_out(b, f  , i+1, j+1)];

						pool_r0c0_f1 = (pool_t->data)[ind_pool_out(b, f+1, i  , j  )]; 
						pool_r0c1_f1 = (pool_t->data)[ind_pool_out(b, f+1, i  , j+1)];
						pool_r1c0_f1 = (pool_t->data)[ind_pool_out(b, f+1, i+1, j  )];
						pool_r1c1_f1 = (pool_t->data)[ind_pool_out(b, f+1, i+1, j+1)];

						pool_r0c0_f2 = (pool_t->data)[ind_pool_out(b, f+2, i  , j  )]; 
						pool_r0c1_f2 = (pool_t->data)[ind_pool_out(b, f+2, i  , j+1)];
						pool_r1c0_f2 = (pool_t->data)[ind_pool_out(b, f+2, i+1, j  )];
						pool_r1c1_f2 = (pool_t->data)[ind_pool_out(b, f+2, i+1, j+1)];

						w_r0c0_f0 = (fully_con_w->data)[ind_fully_con_w(d, f  , i  , j  )];
						w_r0c1_f0 = (fully_con_w->data)[ind_fully_con_w(d, f  , i  , j+1)];
						w_r1c0_f0 = (fully_con_w->data)[ind_fully_con_w(d, f  , i+1, j  )];
						w_r1c1_f0 = (fully_con_w->data)[ind_fully_con_w(d, f  , i+1, j  )];

						w_r0c0_f1 = (fully_con_w->data)[ind_fully_con_w(d, f+1, i  , j  )];
						w_r0c1_f1 = (fully_con_w->data)[ind_fully_con_w(d, f+1, i  , j+1)];
						w_r1c0_f1 = (fully_con_w->data)[ind_fully_con_w(d, f+1, i+1, j  )];
						w_r1c1_f1 = (fully_con_w->data)[ind_fully_con_w(d, f+1, i+1, j  )];

						w_r0c0_f2 = (fully_con_w->data)[ind_fully_con_w(d, f+2, i  , j  )];
						w_r0c1_f2 = (fully_con_w->data)[ind_fully_con_w(d, f+2, i  , j+1)];
						w_r1c0_f2 = (fully_con_w->data)[ind_fully_con_w(d, f+2, i+1, j  )];
						w_r1c1_f2 = (fully_con_w->data)[ind_fully_con_w(d, f+2, i+1, j  )];


						sum_r0c0_f0 += pool_r0c0_f0 * w_r0c0_f0;
						sum_r0c1_f0 += pool_r0c1_f0 * w_r0c1_f0;
						sum_r1c0_f0 += pool_r1c0_f0 * w_r1c0_f0;
						sum_r1c1_f0 += pool_r1c1_f0 * w_r1c1_f0;

						sum_r0c0_f1 += pool_r0c0_f1 * w_r0c0_f1;
						sum_r0c1_f1 += pool_r0c1_f1 * w_r0c1_f1;
						sum_r1c0_f1 += pool_r1c0_f1 * w_r1c0_f1;
						sum_r1c1_f1 += pool_r1c1_f1 * w_r1c1_f1;

						sum_r0c0_f2 += pool_r0c0_f2 * w_r0c0_f2;
						sum_r0c1_f2 += pool_r0c1_f2 * w_r0c1_f2;
						sum_r1c0_f2 += pool_r1c0_f2 * w_r1c0_f2;
						sum_r1c1_f2 += pool_r1c1_f2 * w_r1c1_f2;
					}
				
				}
			}

			// leftover filters not divisible by 3
			for (; f < NUM_FILS; ++f)
			{
				for (int i = 0; i+1 < N_ROWS_POOL; i=i+2)
				{
					for (int j = 0; j+1 < N_COLS_POOL; j=j+2)
					{
						INCREMENT_FLOPS(8)
						pool_r0c0 = (pool_t->data)[ind_pool_out(b, f, i  , j  )]; 
						pool_r0c1 = (pool_t->data)[ind_pool_out(b, f, i  , j+1)];
						pool_r1c0 = (pool_t->data)[ind_pool_out(b, f, i+1, j  )];
						pool_r1c1 = (pool_t->data)[ind_pool_out(b, f, i+1, j+1)];

						w_r0c0 = (fully_con_w->data)[ind_fully_con_w(d, f, i  , j  )];
						w_r0c1 = (fully_con_w->data)[ind_fully_con_w(d, f, i  , j+1)];
						w_r1c0 = (fully_con_w->data)[ind_fully_con_w(d, f, i+1, j  )];
						w_r1c1 = (fully_con_w->data)[ind_fully_con_w(d, f, i+1, j  )];


						sum_r0c0 += pool_r0c0 * w_r0c0;
						sum_r0c1 += pool_r0c1 * w_r0c1;
						sum_r1c0 += pool_r1c0 * w_r1c0;
						sum_r1c1 += pool_r1c1 * w_r1c1;
					}
				
				}
			}
	
			INCREMENT_FLOPS(16)

			sum_1  = sum_r0c0_f0 + sum_r0c1_f0;
			sum_2  = sum_r1c0_f0 + sum_r1c1_f0;
			sum_f0 =       sum_1 + sum_2;

			sum_3  = sum_r0c0_f1 + sum_r0c1_f1;
			sum_4  = sum_r1c0_f1 + sum_r1c1_f1;
			sum_f1 =       sum_3 + sum_4;

			sum_5  = sum_r0c0_f2 + sum_r0c1_f2;
			sum_6  = sum_r1c0_f2 + sum_r1c1_f2;
			sum_f2 =       sum_5 + sum_6;

			sum_7  = sum_r0c0    + sum_r0c1;
			sum_8  = sum_r1c0    + sum_r1c1;
			sum_rem=       sum_7 + sum_8;

			sum_9  = sum_f0 + sum_f1;
			sum_10 = sum_f2 + sum_rem;

			sum = sum_9 + sum_10;

			sum += (fully_con_b->data)[d];

			(fully_con_out->data)[ind_fully_con_out(b, d)] = sum;
		}
	}
}

void softmax(tensor* fully_con_out, tensor* softmax_out, int preds[], int batch_size)
{
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
