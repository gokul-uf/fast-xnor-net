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

// loops on pool rows and cols unrolled, loop on number of filters also unrolled
/*void feed_forward(tensor* pool_t, tensor* fully_con_out, tensor* fully_con_w, tensor* fully_con_b, int batch_size)
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
}*/

// Vectorized
void feed_forward(tensor* pool_t, tensor* fully_con_out, tensor* fully_con_w, tensor* fully_con_b, int batch_size)
{
	double sum;

	int i, j, d;

	__m256d sum_p;

	__m256d sum_d0_p;
	__m256d sum_d1_p;
	__m256d sum_d2_p;
	__m256d sum_d3_p;

	__m256d sum_i0_p;
	__m256d sum_i1_p;
	__m256d sum_i2_p;
	__m256d sum_i3_p;


	__m256d sum_i0_d0_p;
	__m256d sum_i1_d0_p;
	__m256d sum_i2_d0_p;
	__m256d sum_i3_d0_p;

	__m256d sum_i0_d1_p;
	__m256d sum_i1_d1_p;
	__m256d sum_i2_d1_p;
	__m256d sum_i3_d1_p;

	__m256d sum_i0_d2_p;
	__m256d sum_i1_d2_p;
	__m256d sum_i2_d2_p;
	__m256d sum_i3_d2_p;

	__m256d sum_i0_d3_p;
	__m256d sum_i1_d3_p;
	__m256d sum_i2_d3_p;
	__m256d sum_i3_d3_p;


	__m256d pool_i0_p;
	__m256d pool_i1_p;
	__m256d pool_i2_p;
	__m256d pool_i3_p;


	__m256d fully_con_w_i0_p;
	__m256d fully_con_w_i1_p;
	__m256d fully_con_w_i2_p;
	__m256d fully_con_w_i3_p;


	__m256d fully_con_w_i0_d0_p;
	__m256d fully_con_w_i1_d0_p;
	__m256d fully_con_w_i2_d0_p;
	__m256d fully_con_w_i3_d0_p;

	__m256d fully_con_w_i0_d1_p;
	__m256d fully_con_w_i1_d1_p;
	__m256d fully_con_w_i2_d1_p;
	__m256d fully_con_w_i3_d1_p;

	__m256d fully_con_w_i0_d2_p;
	__m256d fully_con_w_i1_d2_p;
	__m256d fully_con_w_i2_d2_p;
	__m256d fully_con_w_i3_d2_p;

	__m256d fully_con_w_i0_d3_p;
	__m256d fully_con_w_i1_d3_p;
	__m256d fully_con_w_i2_d3_p;
	__m256d fully_con_w_i3_d3_p;

	__m256d sum_1_p;
	__m256d sum_2_p;
	__m256d perm_sum_1_p;
	__m256d perm_sum_2_p;
	__m256d bias_p;

	for (int b = 0; b < batch_size; ++b)
	{
		for (d = 0; d+3 < N_DIGS; d=d+4)
		{
			bias_p = _mm256_loadu_pd((fully_con_b->data) + d);

			sum_i0_d0_p = _mm256_set1_pd(0);
			sum_i1_d0_p = _mm256_set1_pd(0);
			sum_i2_d0_p = _mm256_set1_pd(0);
			sum_i3_d0_p = _mm256_set1_pd(0);

			sum_i0_d1_p = _mm256_set1_pd(0);
			sum_i1_d1_p = _mm256_set1_pd(0);
			sum_i2_d1_p = _mm256_set1_pd(0);
			sum_i3_d1_p = _mm256_set1_pd(0);


			sum_i0_d2_p = _mm256_set1_pd(0);
			sum_i1_d2_p = _mm256_set1_pd(0);
			sum_i2_d2_p = _mm256_set1_pd(0);
			sum_i3_d2_p = _mm256_set1_pd(0);

			sum_i0_d3_p = _mm256_set1_pd(0);
			sum_i1_d3_p = _mm256_set1_pd(0);
			sum_i2_d3_p = _mm256_set1_pd(0);
			sum_i3_d3_p = _mm256_set1_pd(0);


			for (int f = 0; f < NUM_FILS; ++f)
			{
				for (i = 0; i+3 < N_ROWS_POOL; i=i+4)
				{
					for (j = 0; j+3 < N_COLS_POOL; j=j+4)
					{
						INCREMENT_FLOPS(2)

						pool_i0_p        = _mm256_loadu_pd( (pool_t->data)      +    ind_pool_out( b, f, i  , j ) );
						pool_i1_p        = _mm256_loadu_pd( (pool_t->data)      +    ind_pool_out( b, f, i+1, j ) );
						pool_i2_p        = _mm256_loadu_pd( (pool_t->data)      +    ind_pool_out( b, f, i+2, j ) );
						pool_i3_p        = _mm256_loadu_pd( (pool_t->data)      +    ind_pool_out( b, f, i+3, j ) );



						fully_con_w_i0_d0_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d  , f, i  , j ) );
						fully_con_w_i1_d0_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d  , f, i+1, j ) );
						fully_con_w_i2_d0_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d  , f, i+2, j ) );
						fully_con_w_i3_d0_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d  , f, i+3, j ) );

						fully_con_w_i0_d1_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+1, f, i  , j ) );
						fully_con_w_i1_d1_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+1, f, i+1, j ) );
						fully_con_w_i2_d1_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+1, f, i+2, j ) );
						fully_con_w_i3_d1_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+1, f, i+3, j ) );

						fully_con_w_i0_d2_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+2, f, i  , j ) );
						fully_con_w_i1_d2_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+2, f, i+1, j ) );
						fully_con_w_i2_d2_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+2, f, i+2, j ) );
						fully_con_w_i3_d2_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+2, f, i+3, j ) );

						fully_con_w_i0_d3_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+3, f, i  , j ) );
						fully_con_w_i1_d3_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+3, f, i+1, j ) );
						fully_con_w_i2_d3_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+3, f, i+2, j ) );
						fully_con_w_i3_d3_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d+3, f, i+3, j ) );



						sum_i0_d0_p = _mm256_add_pd( sum_i0_d0_p, _mm256_mul_pd( pool_i0_p, fully_con_w_i0_d0_p ) );
						sum_i1_d0_p = _mm256_add_pd( sum_i1_d0_p, _mm256_mul_pd( pool_i1_p, fully_con_w_i1_d0_p ) );
						sum_i2_d0_p = _mm256_add_pd( sum_i2_d0_p, _mm256_mul_pd( pool_i2_p, fully_con_w_i2_d0_p ) );
						sum_i3_d0_p = _mm256_add_pd( sum_i3_d0_p, _mm256_mul_pd( pool_i3_p, fully_con_w_i3_d0_p ) );

						sum_i0_d1_p = _mm256_add_pd( sum_i0_d1_p, _mm256_mul_pd( pool_i0_p, fully_con_w_i0_d1_p ) );
						sum_i1_d1_p = _mm256_add_pd( sum_i1_d1_p, _mm256_mul_pd( pool_i1_p, fully_con_w_i1_d1_p ) );
						sum_i2_d1_p = _mm256_add_pd( sum_i2_d1_p, _mm256_mul_pd( pool_i2_p, fully_con_w_i2_d1_p ) );
						sum_i3_d1_p = _mm256_add_pd( sum_i3_d1_p, _mm256_mul_pd( pool_i3_p, fully_con_w_i3_d1_p ) );

						sum_i0_d2_p = _mm256_add_pd( sum_i0_d2_p, _mm256_mul_pd( pool_i0_p, fully_con_w_i0_d2_p ) );
						sum_i1_d2_p = _mm256_add_pd( sum_i1_d2_p, _mm256_mul_pd( pool_i1_p, fully_con_w_i1_d2_p ) );
						sum_i2_d2_p = _mm256_add_pd( sum_i2_d2_p, _mm256_mul_pd( pool_i2_p, fully_con_w_i2_d2_p ) );
						sum_i3_d2_p = _mm256_add_pd( sum_i3_d2_p, _mm256_mul_pd( pool_i3_p, fully_con_w_i3_d2_p ) );

						sum_i0_d3_p = _mm256_add_pd( sum_i0_d3_p, _mm256_mul_pd( pool_i0_p, fully_con_w_i0_d3_p ) );
						sum_i1_d3_p = _mm256_add_pd( sum_i1_d3_p, _mm256_mul_pd( pool_i1_p, fully_con_w_i1_d3_p ) );
						sum_i2_d3_p = _mm256_add_pd( sum_i2_d3_p, _mm256_mul_pd( pool_i2_p, fully_con_w_i2_d3_p ) );
						sum_i3_d3_p = _mm256_add_pd( sum_i3_d3_p, _mm256_mul_pd( pool_i3_p, fully_con_w_i3_d3_p ) );
					}
				}
			}

			sum_d0_p = _mm256_add_pd( _mm256_add_pd( sum_i0_d0_p, sum_i1_d0_p ), _mm256_add_pd( sum_i2_d0_p, sum_i3_d0_p ) );
			sum_d1_p = _mm256_add_pd( _mm256_add_pd( sum_i0_d1_p, sum_i1_d1_p ), _mm256_add_pd( sum_i2_d1_p, sum_i3_d1_p ) );
			sum_d2_p = _mm256_add_pd( _mm256_add_pd( sum_i0_d2_p, sum_i1_d2_p ), _mm256_add_pd( sum_i2_d2_p, sum_i3_d2_p ) );
			sum_d3_p = _mm256_add_pd( _mm256_add_pd( sum_i0_d3_p, sum_i1_d3_p ), _mm256_add_pd( sum_i2_d3_p, sum_i3_d3_p ) );

			// transpose the four vectors
			sum_1_p = _mm256_hadd_pd(sum_d0_p, sum_d2_p);
			sum_2_p = _mm256_hadd_pd(sum_d1_p, sum_d3_p);

			perm_sum_1_p = _mm256_permute4x64_pd(sum_1_p, _MM_SHUFFLE(3,1,2,0));
			perm_sum_2_p = _mm256_permute4x64_pd(sum_2_p, _MM_SHUFFLE(3,1,2,0));


			sum_p = _mm256_hadd_pd( perm_sum_1_p, perm_sum_2_p );
			sum_p = _mm256_add_pd ( sum_p, bias_p);

			_mm256_storeu_pd( (fully_con_out->data) + ind_fully_con_out(b, d  ), sum_p );
		}

		for (; d < N_DIGS; ++d)
		{
			sum_i0_p = _mm256_set1_pd(0);
			sum_i1_p = _mm256_set1_pd(0);
			sum_i2_p = _mm256_set1_pd(0);
			sum_i3_p = _mm256_set1_pd(0);


			for (int f = 0; f < NUM_FILS; ++f)
			{
				for (i = 0; i+3 < N_ROWS_POOL; i=i+4)
				{
					for (j = 0; j+3 < N_COLS_POOL; j=j+4)
					{
						INCREMENT_FLOPS(2)

						pool_i0_p        = _mm256_loadu_pd( (pool_t->data)      +    ind_pool_out( b, f, i  , j ) );
						pool_i1_p        = _mm256_loadu_pd( (pool_t->data)      +    ind_pool_out( b, f, i+1, j ) );
						pool_i2_p        = _mm256_loadu_pd( (pool_t->data)      +    ind_pool_out( b, f, i+2, j ) );
						pool_i3_p        = _mm256_loadu_pd( (pool_t->data)      +    ind_pool_out( b, f, i+3, j ) );

						fully_con_w_i0_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d, f, i  , j ) );
						fully_con_w_i1_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d, f, i+1, j ) );
						fully_con_w_i2_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d, f, i+2, j ) );
						fully_con_w_i3_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w( d, f, i+3, j ) );

						sum_i0_p = _mm256_add_pd( sum_i0_p, _mm256_mul_pd( pool_i0_p, fully_con_w_i0_p ) );
						sum_i1_p = _mm256_add_pd( sum_i1_p, _mm256_mul_pd( pool_i1_p, fully_con_w_i1_p ) );
						sum_i2_p = _mm256_add_pd( sum_i2_p, _mm256_mul_pd( pool_i2_p, fully_con_w_i2_p ) );
						sum_i3_p = _mm256_add_pd( sum_i3_p, _mm256_mul_pd( pool_i3_p, fully_con_w_i3_p ) );
					}
				}
			}

			sum_p = _mm256_add_pd( _mm256_add_pd( sum_i0_p, sum_i1_p ), _mm256_add_pd( sum_i2_p, sum_i3_p ) );

			(fully_con_out->data)[ind_fully_con_out(b, d)] = sum_p[0] + sum_p[1] + sum_p[2] + sum_p[3] + (fully_con_b->data)[d];
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
