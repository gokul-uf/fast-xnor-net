#include "back_prop.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int N_ROWS_POOL;
int N_COLS_POOL;
int TOTAL_FLOPS;
double MULTIPLIER;

// batches in innermost loop
/*void update_sotmax_weights(tensor* fully_con_w, tensor* softmax_out, tensor* pool_t, int* labels, int base, int shuffle_index[])
{
	double delta_w0, delta_w1, delta_w2, delta_w3, delta_w4, delta_w5, delta_w6, delta_w7, delta_w8, delta_w9;
	double delta0, delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta9;
	double mat_val;
	int true_label;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		for (int r = 0; r < N_ROWS_POOL; ++r)
		{
			for (int c = 0; c < N_COLS_POOL; ++c)
			{
				delta_w0 = 0.0, delta_w1 = 0.0, delta_w2 = 0.0, delta_w3 = 0.0, delta_w4 = 0.0, delta_w5 = 0.0,
				delta_w6 = 0.0, delta_w7 = 0.0, delta_w8 = 0.0, delta_w9 = 0.0;
				for (int b = 0; b < BATCH_SIZE; ++b)
				{
					INCREMENT_FLOPS(30)

					// unroll inner loop completely for number of digits

					mat_val = (pool_t->data)[offset(pool_t, b, c, r, f)];
					true_label = labels[shuffle_index[base+b]];

					delta0 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 0)] - (true_label == 0);
					delta1 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 1)] - (true_label == 1);
					delta2 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 2)] - (true_label == 2);
					delta3 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 3)] - (true_label == 3);
					delta4 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 4)] - (true_label == 4);
					delta5 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 5)] - (true_label == 5);
					delta6 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 6)] - (true_label == 6);
					delta7 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 7)] - (true_label == 7);
					delta8 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 8)] - (true_label == 8);
					delta9 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 9)] - (true_label == 9);

					delta_w0 += delta0 * mat_val;
					delta_w1 += delta1 * mat_val;
					delta_w2 += delta2 * mat_val;
					delta_w3 += delta3 * mat_val;
					delta_w4 += delta4 * mat_val;
					delta_w5 += delta5 * mat_val;
					delta_w6 += delta6 * mat_val;
					delta_w7 += delta7 * mat_val;
					delta_w8 += delta8 * mat_val;
					delta_w9 += delta9 * mat_val;
				}

				INCREMENT_FLOPS(20)
				(fully_con_w->data)[offset(fully_con_w, 0, c, r, f)] -= MULTIPLIER*delta_w0;
				(fully_con_w->data)[offset(fully_con_w, 1, c, r, f)] -= MULTIPLIER*delta_w1;
				(fully_con_w->data)[offset(fully_con_w, 2, c, r, f)] -= MULTIPLIER*delta_w2;
				(fully_con_w->data)[offset(fully_con_w, 3, c, r, f)] -= MULTIPLIER*delta_w3;
				(fully_con_w->data)[offset(fully_con_w, 4, c, r, f)] -= MULTIPLIER*delta_w4;
				(fully_con_w->data)[offset(fully_con_w, 5, c, r, f)] -= MULTIPLIER*delta_w5;
				(fully_con_w->data)[offset(fully_con_w, 6, c, r, f)] -= MULTIPLIER*delta_w6;
				(fully_con_w->data)[offset(fully_con_w, 7, c, r, f)] -= MULTIPLIER*delta_w7;
				(fully_con_w->data)[offset(fully_con_w, 8, c, r, f)] -= MULTIPLIER*delta_w8;
				(fully_con_w->data)[offset(fully_con_w, 9, c, r, f)] -= MULTIPLIER*delta_w9;
			}
		}
	}
}*/

// batches in outermost loop
/*void update_sotmax_weights(tensor* fully_con_w, tensor* softmax_out, tensor* pool_t, int* labels, int base, int shuffle_index[])
{
	int true_label;

	double softmax_out_0;
	double softmax_out_1;
	double softmax_out_2;
	double softmax_out_3;
	double softmax_out_4;
	double softmax_out_5;
	double softmax_out_6;
	double softmax_out_7;
	double softmax_out_8;
	double softmax_out_9;

	double delta0;
	double delta1;
	double delta2;
	double delta3;
	double delta4;
	double delta5;
	double delta6;
	double delta7;
	double delta8;
	double delta9;

	double mat_val_f0, mat_val_f1, mat_val_f2;


	double delta_ws[N_ROWS_POOL][N_COLS_POOL][NUM_FILS][N_DIGS];

	// Initialize deltas to zeroes
	for (int r = 0; r < N_ROWS_POOL; ++r)
	{
		for (int c = 0; c < N_COLS_POOL; ++c)
		{
			for (int f = 0; f < NUM_FILS; ++f)
			{
				for (int d = 0; d < N_DIGS; ++d)
				{
					delta_ws[r][c][f][d] = 0.0;
				}
			}
		}
	}

	for (int b = 0; b < BATCH_SIZE; ++b)
	{

		true_label = labels[shuffle_index[base+b]];

		softmax_out_0 = (softmax_out->data)[ind_softmax_out(b, 0)];
		softmax_out_1 = (softmax_out->data)[ind_softmax_out(b, 1)];
		softmax_out_2 = (softmax_out->data)[ind_softmax_out(b, 2)];
		softmax_out_3 = (softmax_out->data)[ind_softmax_out(b, 3)];
		softmax_out_4 = (softmax_out->data)[ind_softmax_out(b, 4)];
		softmax_out_5 = (softmax_out->data)[ind_softmax_out(b, 5)];
		softmax_out_6 = (softmax_out->data)[ind_softmax_out(b, 6)];
		softmax_out_7 = (softmax_out->data)[ind_softmax_out(b, 7)];
		softmax_out_8 = (softmax_out->data)[ind_softmax_out(b, 8)];
		softmax_out_9 = (softmax_out->data)[ind_softmax_out(b, 9)];

		delta0 = softmax_out_0 - (true_label == 0);
		delta1 = softmax_out_1 - (true_label == 1);
		delta2 = softmax_out_2 - (true_label == 2);
		delta3 = softmax_out_3 - (true_label == 3);
		delta4 = softmax_out_4 - (true_label == 4);
		delta5 = softmax_out_5 - (true_label == 5);
		delta6 = softmax_out_6 - (true_label == 6);
		delta7 = softmax_out_7 - (true_label == 7);
		delta8 = softmax_out_8 - (true_label == 8);
		delta9 = softmax_out_9 - (true_label == 9);

		for (int r = 0; r < N_ROWS_POOL; ++r)
		{
			for (int c = 0; c < N_COLS_POOL; ++c)
			{
				INCREMENT_FLOPS(60)

				mat_val_f0 = (pool_t->data)[ind_pool_out(b, 0, r, c)];
				mat_val_f1 = (pool_t->data)[ind_pool_out(b, 1, r, c)];
				mat_val_f2 = (pool_t->data)[ind_pool_out(b, 2, r, c)];

				delta_ws[r][c][0][0] += delta0 * mat_val_f0;
				delta_ws[r][c][0][1] += delta1 * mat_val_f0;
				delta_ws[r][c][0][2] += delta2 * mat_val_f0;
				delta_ws[r][c][0][3] += delta3 * mat_val_f0;
				delta_ws[r][c][0][4] += delta4 * mat_val_f0;
				delta_ws[r][c][0][5] += delta5 * mat_val_f0;
				delta_ws[r][c][0][6] += delta6 * mat_val_f0;
				delta_ws[r][c][0][7] += delta7 * mat_val_f0;
				delta_ws[r][c][0][8] += delta8 * mat_val_f0;
				delta_ws[r][c][0][9] += delta9 * mat_val_f0;

				delta_ws[r][c][1][0] += delta0 * mat_val_f1;
				delta_ws[r][c][1][1] += delta1 * mat_val_f1;
				delta_ws[r][c][1][2] += delta2 * mat_val_f1;
				delta_ws[r][c][1][3] += delta3 * mat_val_f1;
				delta_ws[r][c][1][4] += delta4 * mat_val_f1;
				delta_ws[r][c][1][5] += delta5 * mat_val_f1;
				delta_ws[r][c][1][6] += delta6 * mat_val_f1;
				delta_ws[r][c][1][7] += delta7 * mat_val_f1;
				delta_ws[r][c][1][8] += delta8 * mat_val_f1;
				delta_ws[r][c][1][9] += delta9 * mat_val_f1;

				delta_ws[r][c][2][0] += delta0 * mat_val_f2;
				delta_ws[r][c][2][1] += delta1 * mat_val_f2;
				delta_ws[r][c][2][2] += delta2 * mat_val_f2;
				delta_ws[r][c][2][3] += delta3 * mat_val_f2;
				delta_ws[r][c][2][4] += delta4 * mat_val_f2;
				delta_ws[r][c][2][5] += delta5 * mat_val_f2;
				delta_ws[r][c][2][6] += delta6 * mat_val_f2;
				delta_ws[r][c][2][7] += delta7 * mat_val_f2;
				delta_ws[r][c][2][8] += delta8 * mat_val_f2;
				delta_ws[r][c][2][9] += delta9 * mat_val_f2;

			}
		}
	}

	// update the weights
	for (int r = 0; r < N_ROWS_POOL; ++r)
	{
		for (int c = 0; c < N_COLS_POOL; ++c)
		{

				INCREMENT_FLOPS(60)

				// ---------------------------------------Filter 0-------------------------------------------
				(fully_con_w->data)[ind_fully_con_w(0, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][0];
				(fully_con_w->data)[ind_fully_con_w(1, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][1];
				(fully_con_w->data)[ind_fully_con_w(2, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][2];
				(fully_con_w->data)[ind_fully_con_w(3, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][3];
				(fully_con_w->data)[ind_fully_con_w(4, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][4];
				(fully_con_w->data)[ind_fully_con_w(5, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][5];
				(fully_con_w->data)[ind_fully_con_w(6, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][6];
				(fully_con_w->data)[ind_fully_con_w(7, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][7];
				(fully_con_w->data)[ind_fully_con_w(8, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][8];
				(fully_con_w->data)[ind_fully_con_w(9, 0, r, c)] -= MULTIPLIER * delta_ws[r][c][0][9];

				// ---------------------------------------Filter 1-------------------------------------------
				(fully_con_w->data)[ind_fully_con_w(0, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][0];
				(fully_con_w->data)[ind_fully_con_w(1, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][1];
				(fully_con_w->data)[ind_fully_con_w(2, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][2];
				(fully_con_w->data)[ind_fully_con_w(3, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][3];
				(fully_con_w->data)[ind_fully_con_w(4, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][4];
				(fully_con_w->data)[ind_fully_con_w(5, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][5];
				(fully_con_w->data)[ind_fully_con_w(6, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][6];
				(fully_con_w->data)[ind_fully_con_w(7, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][7];
				(fully_con_w->data)[ind_fully_con_w(8, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][8];
				(fully_con_w->data)[ind_fully_con_w(9, 1, r, c)] -= MULTIPLIER * delta_ws[r][c][1][9];

				// ---------------------------------------Filter 2-------------------------------------------
				(fully_con_w->data)[ind_fully_con_w(0, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][0];
				(fully_con_w->data)[ind_fully_con_w(1, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][1];
				(fully_con_w->data)[ind_fully_con_w(2, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][2];
				(fully_con_w->data)[ind_fully_con_w(3, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][3];
				(fully_con_w->data)[ind_fully_con_w(4, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][4];
				(fully_con_w->data)[ind_fully_con_w(5, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][5];
				(fully_con_w->data)[ind_fully_con_w(6, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][6];
				(fully_con_w->data)[ind_fully_con_w(7, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][7];
				(fully_con_w->data)[ind_fully_con_w(8, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][8];
				(fully_con_w->data)[ind_fully_con_w(9, 2, r, c)] -= MULTIPLIER * delta_ws[r][c][2][9];
		}
	}
}*/

//Vectorized: batches in outermost loop
/*void update_sotmax_weights(tensor* fully_con_w, tensor* softmax_out, tensor* pool_t, int* labels, int base, int shuffle_index[])
{
	int true_label;

	double mat_val_f0, mat_val_f1, mat_val_f2;

	// access in this order
	double* delta_ws = calloc(N_ROWS_POOL * N_COLS_POOL * NUM_FILS * N_DIGS, sizeof(double));

	int index_0_f0, index_1_f0, index_2_f0, index_base_d, index_base_f, index_base_r, index_base_c;
	int index_0_f1, index_1_f1, index_2_f1;
	int index_0_f2, index_1_f2, index_2_f2;

	__m256d multiplier_p = _mm256_set1_pd(MULTIPLIER);

	__m256d ones_p       = _mm256_set1_pd(1);

	__m256d true_label_p;

	__m256d softmax_out_0_p, softmax_out_1_p, softmax_out_2_p;

	__m256d d_0_p, d_1_p, d_2_p;

	__m256d vmask_0, vmask_1, vmask_2;

	__m256d delta_0_p, delta_1_p, delta_2_p;


	__m256d delta_ws_0_r0_p;
	__m256d delta_ws_1_r1_p;
	__m256d delta_ws_2_r2_p;
	__m256d delta_ws_3_r3_p;

	__m256d delta_ws_0_f0_p;

	__m256d delta_ws_0_f1_p;
	__m256d delta_ws_1_f1_p;
	__m256d delta_ws_2_f1_p;

	__m256d delta_ws_0_f2_p;
	__m256d delta_ws_1_f2_p;
	__m256d delta_ws_2_f2_p;

	__m256d mat_val_r0_p;
	__m256d mat_val_r1_p;
	__m256d mat_val_r2_p;
	__m256d mat_val_r3_p;

	__m256d w_0_f0_p;
	__m256d w_1_f0_p;
	__m256d w_2_f0_p;

	__m256d w_0_f1_p;
	__m256d w_1_f1_p;
	__m256d w_2_f1_p;

	__m256d w_0_f2_p;
	__m256d w_1_f2_p;
	__m256d w_2_f2_p;

	__m256d and_p;


	for (int b = 0; b < BATCH_SIZE; ++b)
	{

		true_label   = labels[shuffle_index[base+b]];
		true_label_p = _mm256_set1_pd(true_label);

		for (int d = 0; d < N_DIGS; ++d)
		{

			index_base_d = ( d * NUM_FILS*N_ROWS_POOL*N_COLS_POOL );

			vmask_0 = _mm256_cmp_pd(   true_label_p, _mm256_set1_pd(d), 0x10); // 0x10 => EQ.
			and_p   = _mm256_and_pd(ones_p, vmask_0);

			softmax_out_0_p = _mm256_set1_pd( (softmax_out->data)[ind_softmax_out(b, d)] );
			delta_0_p       = _mm256_sub_pd(softmax_out_0_p, and_p);

			for (int f = 0; f < NUM_FILS; ++f)
			{

				index_base_f = index_base_d + ( f * N_ROWS_POOL*N_COLS_POOL );

				for (int r = 0; r+3 < N_ROWS_POOL; r=r+4)
				{

					int index_base_r0 = index_base_f + ( (r  ) * N_COLS_POOL);
					int index_base_r1 = index_base_f + ( (r+1) * N_COLS_POOL);
					int index_base_r2 = index_base_f + ( (r+2) * N_COLS_POOL);
					int index_base_r3 = index_base_f + ( (r+3) * N_COLS_POOL);

					for (int c = 0; c+3 < N_COLS_POOL; c=c+4)
					{
						INCREMENT_FLOPS(60)

						int index_r0_c = index_base_r0 + c;
						int index_r1_c = index_base_r1 + c;
						int index_r2_c = index_base_r2 + c;
						int index_r3_c = index_base_r3 + c;

						delta_ws_0_r0_p = _mm256_loadu_pd( delta_ws + index_r0_c);
						delta_ws_1_r1_p = _mm256_loadu_pd( delta_ws + index_r1_c);
						delta_ws_2_r2_p = _mm256_loadu_pd( delta_ws + index_r2_c);
						delta_ws_3_r3_p = _mm256_loadu_pd( delta_ws + index_r3_c);

						mat_val_r0_p    = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r  , c) );
						mat_val_r1_p    = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r+1, c) );
						mat_val_r2_p    = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r+2, c) );
						mat_val_r3_p    = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r+3, c) );

						delta_ws_0_r0_p = _mm256_add_pd( delta_ws_0_r0_p, _mm256_mul_pd( delta_0_p, mat_val_r0_p ) );
						delta_ws_1_r1_p = _mm256_add_pd( delta_ws_1_r1_p, _mm256_mul_pd( delta_0_p, mat_val_r1_p ) );
						delta_ws_2_r2_p = _mm256_add_pd( delta_ws_2_r2_p, _mm256_mul_pd( delta_0_p, mat_val_r2_p ) );
						delta_ws_3_r3_p = _mm256_add_pd( delta_ws_3_r3_p, _mm256_mul_pd( delta_0_p, mat_val_r3_p ) );

						_mm256_storeu_pd(delta_ws + index_r0_c, delta_ws_0_r0_p);
						_mm256_storeu_pd(delta_ws + index_r1_c, delta_ws_1_r1_p);
						_mm256_storeu_pd(delta_ws + index_r2_c, delta_ws_2_r2_p);
						_mm256_storeu_pd(delta_ws + index_r3_c, delta_ws_3_r3_p);
					}
				}
			}
		}
	}

	// update the weights
	for (int d = 0; d < N_DIGS; ++d)
	{

		index_base_d = ( d * NUM_FILS*N_ROWS_POOL*N_COLS_POOL );

		for (int f = 0; f < NUM_FILS; ++f)
		{

			index_base_f = index_base_d + ( f * N_ROWS_POOL*N_COLS_POOL );

			for (int r = 0; r < N_ROWS_POOL; ++r)
			{

				index_base_r = index_base_f + (r * N_COLS_POOL);

				for (int c = 0; c+3 < N_COLS_POOL; c=c+4)
				{

					INCREMENT_FLOPS(60)

					index_0_f0 = index_base_r + c;

					delta_ws_0_f0_p = _mm256_loadu_pd(delta_ws + index_0_f0);

					// ---------------------------------------Filter 0-------------------------------------------

					w_0_f0_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r, c) );


					w_0_f0_p = _mm256_sub_pd( w_0_f0_p, _mm256_mul_pd( multiplier_p, delta_ws_0_f0_p ) );

					_mm256_storeu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r, c), w_0_f0_p );
				}
			}
		}
	}

	free(delta_ws);
}*/

// No unrolling
/*void update_sotmax_weights(tensor* fully_con_w, tensor* softmax_out, tensor* pool_t, int* labels, int base, int shuffle_index[])
{
	int cur_label;

	for (int d = 0; d < N_DIGS; ++d)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					double delta_w = 0.0, delta = 0.0;
					for (int b = 0; b < BATCH_SIZE; ++b)
					{
						INCREMENT_FLOPS(3)

						cur_label = labels[shuffle_index[base+b]];

						delta = softmax_out->data[ind_softmax_out(b, d)] - (cur_label == d);
						delta_w += delta * pool_t->data[ind_pool_out(b, f, r, c)];
					}

					INCREMENT_FLOPS(2)

					(fully_con_w->data)[ind_fully_con_w(d, f, r, c)] -= MULTIPLIER*delta_w;
				}
			}
		}
	}
}*/

//Vectorized: batch in innermoest loop, best
void update_sotmax_weights(tensor* fully_con_w, tensor* softmax_out, tensor* pool_t, int* labels, int base, int shuffle_index[])
{
	int cur_label, c, r;

	__m256d delta_w_p, cur_label_p, delta_p, pool_p, w_p;

	double delta, delta_w, delta_r0_w, delta_r1_w, delta_r2_w, delta_r3_w;

	__m256d multiplier_p = _mm256_set1_pd(MULTIPLIER);

	__m256d pool_r0_p;
	__m256d pool_r1_p;
	__m256d pool_r2_p;
	__m256d pool_r3_p;

	__m256d delta_w_r0_p;
	__m256d delta_w_r1_p;
	__m256d delta_w_r2_p;
	__m256d delta_w_r3_p;

	__m256d w_r0_p;
	__m256d w_r1_p;
	__m256d w_r2_p;
	__m256d w_r3_p;

	int _3_COLS_POOL = 3 * N_COLS_POOL;
	int SIZE_POOL_OUT = BATCH_SIZE*NUM_FILS*N_ROWS_POOL*N_COLS_POOL;
	int ONE_BATCH_POOL_OUT = NUM_FILS*N_ROWS_POOL*N_COLS_POOL;

	int fully_index0 = 0;
	int fully_index1 = N_COLS_POOL;
	int fully_index2 = 2 * N_COLS_POOL;
	int fully_index3 = 3 * N_COLS_POOL;

	double* pool_data = pool_t->data;
	double* fully_con_w_data = fully_con_w->data;

	for (int d = 0; d < N_DIGS; ++d)
	{
		int pool_index0 = 0;
		int pool_index1 = N_COLS_POOL;
		int pool_index2 = 2 * N_COLS_POOL;
		int pool_index3 = 3 * N_COLS_POOL;
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (r = 0; r+3 < N_ROWS_POOL; r=r+4)
			{
				for (c = 0; c+3 < N_COLS_POOL; c=c+4)
				{

					delta_w_r0_p = _mm256_set1_pd(0);
					delta_w_r1_p = _mm256_set1_pd(0);
					delta_w_r2_p = _mm256_set1_pd(0);
					delta_w_r3_p = _mm256_set1_pd(0);

					for (int b = 0; b < BATCH_SIZE; ++b)
					{
						INCREMENT_FLOPS(3)

						cur_label   = labels[shuffle_index[base+b]];
						cur_label_p = _mm256_set1_pd(cur_label);

						delta   = softmax_out->data[ind_softmax_out(b, d)] - (cur_label == d);
						delta_p = _mm256_set1_pd(delta);

						//printf("%d\n", ind_pool_out(b, f, r, c) == pool_index0);
						//printf("%d\n", ind_pool_out(b, f, r+1, c) == pool_index1);
						//printf("%d\n", ind_pool_out(b, f, r+2, c) == pool_index2);
						//printf("%d\n", ind_pool_out(b, f, r+3, c) == pool_index3);
						//pool_r0_p = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r  , c) );
						//pool_r1_p = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r+1, c) );
						//pool_r2_p = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r+2, c) );
						//pool_r3_p = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r+3, c) );
						pool_r0_p = _mm256_loadu_pd( pool_data + pool_index0 );
						pool_r1_p = _mm256_loadu_pd( pool_data + pool_index1 );
						pool_r2_p = _mm256_loadu_pd( pool_data + pool_index2 );
						pool_r3_p = _mm256_loadu_pd( pool_data + pool_index3 );

						delta_w_r0_p = _mm256_add_pd( delta_w_r0_p, _mm256_mul_pd( delta_p, pool_r0_p ) );
						delta_w_r1_p = _mm256_add_pd( delta_w_r1_p, _mm256_mul_pd( delta_p, pool_r1_p ) );
						delta_w_r2_p = _mm256_add_pd( delta_w_r2_p, _mm256_mul_pd( delta_p, pool_r2_p ) );
						delta_w_r3_p = _mm256_add_pd( delta_w_r3_p, _mm256_mul_pd( delta_p, pool_r3_p ) );

						pool_index0 += ONE_BATCH_POOL_OUT;
						pool_index1 += ONE_BATCH_POOL_OUT;
						pool_index2 += ONE_BATCH_POOL_OUT;
						pool_index3 += ONE_BATCH_POOL_OUT;
					}

					pool_index0 -= SIZE_POOL_OUT;
					pool_index1 -= SIZE_POOL_OUT;
					pool_index2 -= SIZE_POOL_OUT;
					pool_index3 -= SIZE_POOL_OUT;

					INCREMENT_FLOPS(2)

					//printf("%d\n", ind_fully_con_w(d, f, r, c) == fully_index0);
					//printf("%d\n", ind_fully_con_w(d, f, r+1, c) == fully_index1);
					//printf("%d\n", ind_fully_con_w(d, f, r+2, c) == fully_index2);
					//printf("%d\n", ind_fully_con_w(d, f, r+3, c) == fully_index3);
					//w_r0_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r  , c) );
					//w_r1_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r+1, c) );
					//w_r2_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r+2, c) );
					//w_r3_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r+3, c) );
					w_r0_p = _mm256_loadu_pd( fully_con_w_data + fully_index0 );
					w_r1_p = _mm256_loadu_pd( fully_con_w_data + fully_index1 );
					w_r2_p = _mm256_loadu_pd( fully_con_w_data + fully_index2 );
					w_r3_p = _mm256_loadu_pd( fully_con_w_data + fully_index3 );

					w_r0_p = _mm256_sub_pd( w_r0_p, _mm256_mul_pd(multiplier_p, delta_w_r0_p));
					w_r1_p = _mm256_sub_pd( w_r1_p, _mm256_mul_pd(multiplier_p, delta_w_r1_p));
					w_r2_p = _mm256_sub_pd( w_r2_p, _mm256_mul_pd(multiplier_p, delta_w_r2_p));
					w_r3_p = _mm256_sub_pd( w_r3_p, _mm256_mul_pd(multiplier_p, delta_w_r3_p));

					//_mm256_storeu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r  , c), w_r0_p );
					//_mm256_storeu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r+1, c), w_r1_p );
					//_mm256_storeu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r+2, c), w_r2_p );
					//_mm256_storeu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r+3, c), w_r3_p );
					_mm256_storeu_pd( fully_con_w_data + fully_index0, w_r0_p );
					_mm256_storeu_pd( fully_con_w_data + fully_index1, w_r1_p );
					_mm256_storeu_pd( fully_con_w_data + fully_index2, w_r2_p );
					_mm256_storeu_pd( fully_con_w_data + fully_index3, w_r3_p );

					pool_index0 += 4;
					pool_index1 += 4;
					pool_index2 += 4;
					pool_index3 += 4;

					fully_index0 += 4;
					fully_index1 += 4;
					fully_index2 += 4;
					fully_index3 += 4;
				}

				for (; c < N_COLS_POOL; ++c)
				{

					delta_r0_w = 0.0, delta_r1_w = 0.0, delta_r2_w, delta_r3_w;

					for (int b = 0; b < BATCH_SIZE; ++b)
					{
						INCREMENT_FLOPS(3)

						cur_label = labels[shuffle_index[base+b]];

						delta = softmax_out->data[ind_softmax_out(b, d)] - (cur_label == d);

						//printf("%d\n", ind_pool_out(b, f, r, c) == pool_index0);
						//printf("%d\n", ind_pool_out(b, f, r+1, c) == pool_index1);
						//printf("%d\n", ind_pool_out(b, f, r+2, c) == pool_index2);
						//printf("%d\n", ind_pool_out(b, f, r+3, c) == pool_index3);
						//delta_r0_w += delta * pool_t->data[ind_pool_out(b, f, r  , c)];
						//delta_r1_w += delta * pool_t->data[ind_pool_out(b, f, r+1, c)];
						//delta_r2_w += delta * pool_t->data[ind_pool_out(b, f, r+2, c)];
						//delta_r3_w += delta * pool_t->data[ind_pool_out(b, f, r+3, c)];
						delta_r0_w += delta * pool_data[pool_index0];
						delta_r1_w += delta * pool_data[pool_index1];
						delta_r2_w += delta * pool_data[pool_index2];
						delta_r3_w += delta * pool_data[pool_index3];

						pool_index0 += ONE_BATCH_POOL_OUT;
						pool_index1 += ONE_BATCH_POOL_OUT;
						pool_index2 += ONE_BATCH_POOL_OUT;
						pool_index3 += ONE_BATCH_POOL_OUT;
					}

					pool_index0 -= SIZE_POOL_OUT;
					pool_index1 -= SIZE_POOL_OUT;
					pool_index2 -= SIZE_POOL_OUT;
					pool_index3 -= SIZE_POOL_OUT;

					INCREMENT_FLOPS(2)

					//printf("%d\n", ind_fully_con_w(d, f, r, c) == fully_index0);
					//printf("%d\n", ind_fully_con_w(d, f, r+1, c) == fully_index1);
					//printf("%d\n", ind_fully_con_w(d, f, r+2, c) == fully_index2);
					//printf("%d\n", ind_fully_con_w(d, f, r+3, c) == fully_index3);
					//(fully_con_w->data)[ind_fully_con_w(d, f, r  , c)] -= MULTIPLIER * delta_r0_w;
					//(fully_con_w->data)[ind_fully_con_w(d, f, r+1, c)] -= MULTIPLIER * delta_r1_w;
					//(fully_con_w->data)[ind_fully_con_w(d, f, r+2, c)] -= MULTIPLIER * delta_r2_w;
					//(fully_con_w->data)[ind_fully_con_w(d, f, r+3, c)] -= MULTIPLIER * delta_r3_w;
					fully_con_w_data[fully_index0] -= MULTIPLIER * delta_r0_w;
					fully_con_w_data[fully_index1] -= MULTIPLIER * delta_r1_w;
					fully_con_w_data[fully_index2] -= MULTIPLIER * delta_r2_w;
					fully_con_w_data[fully_index3] -= MULTIPLIER * delta_r3_w;

					pool_index0 += 1;
					pool_index1 += 1;
					pool_index2 += 1;
					pool_index3 += 1;

					fully_index0 += 1;
					fully_index1 += 1;
					fully_index2 += 1;
					fully_index3 += 1;
				}
				pool_index0 += _3_COLS_POOL;
				pool_index1 += _3_COLS_POOL;
				pool_index2 += _3_COLS_POOL;
				pool_index3 += _3_COLS_POOL;

				fully_index0 += _3_COLS_POOL;
				fully_index1 += _3_COLS_POOL;
				fully_index2 += _3_COLS_POOL;
				fully_index3 += _3_COLS_POOL;
			}

			for (; r < N_ROWS_POOL; ++r)
			{
				for (c = 0; c+3 < N_COLS_POOL; c=c+4)
				{

					delta_w_p = _mm256_set1_pd(0);

					for (int b = 0; b < BATCH_SIZE; ++b)
					{
						INCREMENT_FLOPS(3)

						cur_label   = labels[shuffle_index[base+b]];
						cur_label_p = _mm256_set1_pd(cur_label);

						delta   = softmax_out->data[ind_softmax_out(b, d)] - (cur_label == d);
						delta_p = _mm256_set1_pd(delta);

						//printf("%d\n", ind_pool_out(b, f, r, c) == pool_index0);
						//pool_p    = _mm256_loadu_pd( (pool_t->data) + ind_pool_out(b, f, r, c) );
						pool_p    = _mm256_loadu_pd( (pool_data) + pool_index0 );
						delta_w_p = _mm256_add_pd( delta_w_p, _mm256_mul_pd( delta_p, pool_p ) );

						pool_index0 += ONE_BATCH_POOL_OUT;
					}

					pool_index0 -= SIZE_POOL_OUT;

					INCREMENT_FLOPS(2)

					//w_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r, c) );
					w_p = _mm256_loadu_pd( fully_con_w_data + fully_index0 );
					w_p = _mm256_sub_pd( w_p, _mm256_mul_pd(multiplier_p, delta_w_p));

					//printf("%d\n", ind_fully_con_w(d, f, r, c) == fully_index0);
					//_mm256_storeu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r, c), w_p );
					_mm256_storeu_pd( fully_con_w_data + fully_index0, w_p );

					pool_index0 += 4;
					pool_index1 += 4;
					pool_index2 += 4;
					pool_index3 += 4;

					fully_index0 += 4;
					fully_index1 += 4;
					fully_index2 += 4;
					fully_index3 += 4;
				}

				for (; c < N_COLS_POOL; ++c)
				{

					delta_w = 0.0, delta = 0.0;

					for (int b = 0; b < BATCH_SIZE; ++b)
					{
						INCREMENT_FLOPS(3)

						cur_label = labels[shuffle_index[base+b]];

						delta = softmax_out->data[ind_softmax_out(b, d)] - (cur_label == d);
						delta_w += delta * pool_data[pool_index0];
						//delta_w += delta * pool_t->data[ind_pool_out(b, f, r, c)];
						//printf("%d\n", ind_pool_out(b, f, r, c) == pool_index0);

						pool_index0 += ONE_BATCH_POOL_OUT;
					}

					pool_index0 -= SIZE_POOL_OUT;

					INCREMENT_FLOPS(2)

					//printf("%d\n", ind_fully_con_w(d, f, r, c) == fully_index0);
					//(fully_con_w->data)[ind_fully_con_w(d, f, r, c)] -= MULTIPLIER*delta_w;
					fully_con_w_data[fully_index0] -= MULTIPLIER*delta_w;

					pool_index0 += 1;
					pool_index1 += 1;
					pool_index2 += 1;
					pool_index3 += 1;

					fully_index0 += 1;
					fully_index1 += 1;
					fully_index2 += 1;
					fully_index3 += 1;
				}
			}
		}
	}
}

// loop on digits unrolled
/*void update_sotmax_biases(tensor* fully_con_b, tensor* softmax_out, int* labels, int base, int shuffle_index[])
{
	double delta0=0, delta1=0, delta2=0, delta3=0, delta4=0, delta5=0, delta6=0, delta7=0, delta8=0, delta9=0;
	double delta00=0, delta11=0, delta22=0, delta33=0, delta44=0, delta55=0, delta66=0, delta77=0, delta88=0, delta99=0;

	int true_label1, true_label2;

	for (int b = 0; b < BATCH_SIZE; b=b+2)
	{
		INCREMENT_FLOPS(40)

		true_label1 = labels[shuffle_index[base+b]];

		delta0 += softmax_out->data[ind_softmax_out(b, 0)] - (true_label1 == 0);
		delta1 += softmax_out->data[ind_softmax_out(b, 1)] - (true_label1 == 1);
		delta2 += softmax_out->data[ind_softmax_out(b, 2)] - (true_label1 == 2);
		delta3 += softmax_out->data[ind_softmax_out(b, 3)] - (true_label1 == 3);
		delta4 += softmax_out->data[ind_softmax_out(b, 4)] - (true_label1 == 4);
		delta5 += softmax_out->data[ind_softmax_out(b, 5)] - (true_label1 == 5);
		delta6 += softmax_out->data[ind_softmax_out(b, 6)] - (true_label1 == 6);
		delta7 += softmax_out->data[ind_softmax_out(b, 7)] - (true_label1 == 7);
		delta8 += softmax_out->data[ind_softmax_out(b, 8)] - (true_label1 == 8);
		delta9 += softmax_out->data[ind_softmax_out(b, 9)] - (true_label1 == 9);


		true_label2 = labels[shuffle_index[base+b+1]];

		delta00 += softmax_out->data[ind_softmax_out(b+1, 0)] - (true_label2 == 0);
		delta11 += softmax_out->data[ind_softmax_out(b+1, 1)] - (true_label2 == 1);
		delta22 += softmax_out->data[ind_softmax_out(b+1, 2)] - (true_label2 == 2);
		delta33 += softmax_out->data[ind_softmax_out(b+1, 3)] - (true_label2 == 3);
		delta44 += softmax_out->data[ind_softmax_out(b+1, 4)] - (true_label2 == 4);
		delta55 += softmax_out->data[ind_softmax_out(b+1, 5)] - (true_label2 == 5);
		delta66 += softmax_out->data[ind_softmax_out(b+1, 6)] - (true_label2 == 6);
		delta77 += softmax_out->data[ind_softmax_out(b+1, 7)] - (true_label2 == 7);
		delta88 += softmax_out->data[ind_softmax_out(b+1, 8)] - (true_label2 == 8);
		delta99 += softmax_out->data[ind_softmax_out(b+1, 9)] - (true_label2 == 9);

	}

	INCREMENT_FLOPS(30)
	delta0 += delta00;
	delta1 += delta11;
	delta2 += delta22;
	delta3 += delta33;
	delta4 += delta44;
	delta5 += delta55;
	delta6 += delta66;
	delta7 += delta77;
	delta8 += delta88;
	delta9 += delta99;

	(fully_con_b->data)[0] -= MULTIPLIER*delta0;
	(fully_con_b->data)[1] -= MULTIPLIER*delta1;
	(fully_con_b->data)[2] -= MULTIPLIER*delta2;
	(fully_con_b->data)[3] -= MULTIPLIER*delta3;
	(fully_con_b->data)[4] -= MULTIPLIER*delta4;
	(fully_con_b->data)[5] -= MULTIPLIER*delta5;
	(fully_con_b->data)[6] -= MULTIPLIER*delta6;
	(fully_con_b->data)[7] -= MULTIPLIER*delta7;
	(fully_con_b->data)[8] -= MULTIPLIER*delta8;
	(fully_con_b->data)[9] -= MULTIPLIER*delta9;
}*/

// No unrolling
void update_sotmax_biases(tensor* fully_con_b, tensor* softmax_out, int* labels, int base, int shuffle_index[]){

	for (int d = 0; d < N_DIGS; ++d)
	{
		double delta_b = 0.0, delta = 0.0;
		for (int b = 0; b < BATCH_SIZE; ++b)
		{

			INCREMENT_FLOPS(2)

			delta = softmax_out->data[ind_softmax_out(b, d)] - (labels[shuffle_index[base+b]] == d);
			delta_b += delta;
		}

		INCREMENT_FLOPS(2)

		(fully_con_b->data)[d] -= MULTIPLIER*delta_b;

	}
}

// loop on digits unrolled
/*void bp_softmax_to_maxpool(tensor* del_max_pool, tensor* softmax_out, int* labels, int base, tensor* fully_con_w, int shuffle_index[])
{
	double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9;
	double sum01, sum23, sum45, sum67, sum89;
	double sum0123, sum4567, sum0_7, sum_fin;

	double delta0, delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta9;

	double w0, w1, w2, w3, w4 ,w5, w6, w7, w8, w9;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		int cur_label = labels[shuffle_index[base+b]];
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					// Unrolling the inner loop on digits completely

					delta0 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 0)] - (cur_label == 0);
					delta1 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 1)] - (cur_label == 1);
					delta2 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 2)] - (cur_label == 2);
					delta3 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 3)] - (cur_label == 3);
					delta4 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 4)] - (cur_label == 4);
					delta5 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 5)] - (cur_label == 5);
					delta6 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 6)] - (cur_label == 6);
					delta7 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 7)] - (cur_label == 7);
					delta8 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 8)] - (cur_label == 8);
					delta9 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 9)] - (cur_label == 9);

					w0 = (fully_con_w->data)[offset(fully_con_w, 0, c, r, f)];
					w1 = (fully_con_w->data)[offset(fully_con_w, 1, c, r, f)];
					w2 = (fully_con_w->data)[offset(fully_con_w, 2, c, r, f)];
					w3 = (fully_con_w->data)[offset(fully_con_w, 3, c, r, f)];
					w4 = (fully_con_w->data)[offset(fully_con_w, 4, c, r, f)];
					w5 = (fully_con_w->data)[offset(fully_con_w, 5, c, r, f)];
					w6 = (fully_con_w->data)[offset(fully_con_w, 6, c, r, f)];
					w7 = (fully_con_w->data)[offset(fully_con_w, 7, c, r, f)];
					w8 = (fully_con_w->data)[offset(fully_con_w, 8, c, r, f)];
					w9 = (fully_con_w->data)[offset(fully_con_w, 9, c, r, f)];

					sum0 = delta0 * w0;
					sum1 = delta1 * w1;
					sum2 = delta2 * w2;
					sum3 = delta3 * w3;
					sum4 = delta4 * w4;
					sum5 = delta5 * w5;
					sum6 = delta6 * w6;
					sum7 = delta7 * w7;
					sum8 = delta8 * w8;
					sum9 = delta9 * w9;

					sum01 = sum0+sum1;
					sum23 = sum2+sum3;
					sum45 = sum4+sum5;
					sum67 = sum6+sum7;
					sum89 = sum8+sum9;

					sum0123 = sum01   + sum23;
					sum4567 = sum45   + sum67;
					sum0_7  = sum0123 + sum4567;
					sum_fin = sum0_7  + sum89;

					(del_max_pool->data)[offset(del_max_pool, b, c, r, f)] = sum_fin;
				}
			}
		}
	}
}*/

// No unrolling
void bp_softmax_to_maxpool(tensor* del_max_pool, tensor* softmax_out, int* labels, int base, tensor* fully_con_w, int shuffle_index[]){
	double sum = 0.0;
	double delta = 0.0;
	int cur_label;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{

		cur_label = labels[shuffle_index[base+b]];

		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					sum = 0.0;

					for (int d = 0; d < N_DIGS; ++d)
					{

						INCREMENT_FLOPS(3)

						delta = (softmax_out->data)[ind_softmax_out(b, d)] - (cur_label == d);

						sum += delta * (fully_con_w->data)[ind_fully_con_w(d, f, r, c)];
					}

					(del_max_pool->data)[ind_pool_out(b, f, r, c)] = sum;

				}
			}
		}
	}
}

// No unrolling
void bp_maxpool_to_conv(tensor* del_conv, tensor* del_max_pool, tensor* conv_t, int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL],
   	int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (int c = 0; c < N_COLS_POOL; ++c)
				{

					INCREMENT_FLOPS(1)

					int row = pool_index_i[b][f][r][c];
					int col = pool_index_j[b][f][r][c];

					if ( (conv_t->data)[ind_conv_out(b, f, row, col)] > 0.0)
					{
						(del_conv->data)[ind_conv_out(b, f, row, col)] = (del_max_pool->data)[ind_pool_out(b, f, r, c)];
					}
				}
			}
		}
	}
}

// inner loop on digits unrolled completely
/*void bp_softmax_to_conv(tensor* del_conv, tensor* softmax_out, tensor* conv_t, int* labels, int base, tensor* fully_con_w,
	int shuffle_index[], int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9;
	double sum01, sum23, sum45, sum67, sum89;
	double sum0123, sum4567, sum0_7, sum_fin;

	double delta0, delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta9;

	double w0, w1, w2, w3, w4 ,w5, w6, w7, w8, w9;

	double softmax_out_0;
	double softmax_out_1;
	double softmax_out_2;
	double softmax_out_3;
	double softmax_out_4;
	double softmax_out_5;
	double softmax_out_6;
	double softmax_out_7;
	double softmax_out_8;
	double softmax_out_9;

	int row, col, cur_label;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		cur_label = labels[shuffle_index[base+b]];

		softmax_out_0 = (softmax_out->data)[ind_softmax_out(b, 0)];
		softmax_out_1 = (softmax_out->data)[ind_softmax_out(b, 1)];
		softmax_out_2 = (softmax_out->data)[ind_softmax_out(b, 2)];
		softmax_out_3 = (softmax_out->data)[ind_softmax_out(b, 3)];
		softmax_out_4 = (softmax_out->data)[ind_softmax_out(b, 4)];
		softmax_out_5 = (softmax_out->data)[ind_softmax_out(b, 5)];
		softmax_out_6 = (softmax_out->data)[ind_softmax_out(b, 6)];
		softmax_out_7 = (softmax_out->data)[ind_softmax_out(b, 7)];
		softmax_out_8 = (softmax_out->data)[ind_softmax_out(b, 8)];
		softmax_out_9 = (softmax_out->data)[ind_softmax_out(b, 9)];

		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					INCREMENT_FLOPS(30)
					// Unrolling the inner loop on digits completely

					delta0 = softmax_out_0 - (cur_label == 0);
					delta1 = softmax_out_1 - (cur_label == 1);
					delta2 = softmax_out_2 - (cur_label == 2);
					delta3 = softmax_out_3 - (cur_label == 3);
					delta4 = softmax_out_4 - (cur_label == 4);
					delta5 = softmax_out_5 - (cur_label == 5);
					delta6 = softmax_out_6 - (cur_label == 6);
					delta7 = softmax_out_7 - (cur_label == 7);
					delta8 = softmax_out_8 - (cur_label == 8);
					delta9 = softmax_out_9 - (cur_label == 9);

					w0 = (fully_con_w->data)[ind_fully_con_w(0, f, r, c)];
					w1 = (fully_con_w->data)[ind_fully_con_w(1, f, r, c)];
					w2 = (fully_con_w->data)[ind_fully_con_w(2, f, r, c)];
					w3 = (fully_con_w->data)[ind_fully_con_w(3, f, r, c)];
					w4 = (fully_con_w->data)[ind_fully_con_w(4, f, r, c)];
					w5 = (fully_con_w->data)[ind_fully_con_w(5, f, r, c)];
					w6 = (fully_con_w->data)[ind_fully_con_w(6, f, r, c)];
					w7 = (fully_con_w->data)[ind_fully_con_w(7, f, r, c)];
					w8 = (fully_con_w->data)[ind_fully_con_w(8, f, r, c)];
					w9 = (fully_con_w->data)[ind_fully_con_w(9, f, r, c)];

					sum0 = delta0 * w0;
					sum1 = delta1 * w1;
					sum2 = delta2 * w2;
					sum3 = delta3 * w3;
					sum4 = delta4 * w4;
					sum5 = delta5 * w5;
					sum6 = delta6 * w6;
					sum7 = delta7 * w7;
					sum8 = delta8 * w8;
					sum9 = delta9 * w9;

					sum01 = sum0+sum1;
					sum23 = sum2+sum3;
					sum45 = sum4+sum5;
					sum67 = sum6+sum7;
					sum89 = sum8+sum9;

					sum0123 = sum01   + sum23;
					sum4567 = sum45   + sum67;
					sum0_7  = sum0123 + sum4567;
					sum_fin = sum0_7  + sum89;


					// bp from max_pool to conv
					row = pool_index_i[b][f][r][c];
					col = pool_index_j[b][f][r][c];

					if (conv_t->data[ind_conv_out(b, f, row, col)] > 0.0)
					{
						(del_conv->data)[ind_conv_out(b, f, row, col)] = sum_fin;
					}

				}
			}
		}
	}
}*/

// inner loops on digits and filters unrolled completely
/*void bp_softmax_to_conv(tensor* del_conv, tensor* softmax_out, tensor* conv_t, int* labels, int base, tensor* fully_con_w,
	int shuffle_index[], int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double delta0, delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta9;

	double softmax_out_0;
	double softmax_out_1;
	double softmax_out_2;
	double softmax_out_3;
	double softmax_out_4;
	double softmax_out_5;
	double softmax_out_6;
	double softmax_out_7;
	double softmax_out_8;
	double softmax_out_9;

	double w0_f0, w0_f1, w0_f2;
    double w1_f0, w1_f1, w1_f2;
    double w2_f0, w2_f1, w2_f2;
    double w3_f0, w3_f1, w3_f2;
    double w4_f0, w4_f1, w4_f2;
    double w5_f0, w5_f1, w5_f2;
    double w6_f0, w6_f1, w6_f2;
    double w7_f0, w7_f1, w7_f2;
    double w8_f0, w8_f1, w8_f2;
    double w9_f0, w9_f1, w9_f2;

	double sum0_f0, sum0_f1, sum0_f2;
	double sum1_f0, sum1_f1, sum1_f2;
	double sum2_f0, sum2_f1, sum2_f2;
	double sum3_f0, sum3_f1, sum3_f2;
	double sum4_f0, sum4_f1, sum4_f2;
	double sum5_f0, sum5_f1, sum5_f2;
	double sum6_f0, sum6_f1, sum6_f2;
	double sum7_f0, sum7_f1, sum7_f2;
	double sum8_f0, sum8_f1, sum8_f2;
	double sum9_f0, sum9_f1, sum9_f2;

    double sum01_f0, sum01_f1, sum01_f2;
    double sum23_f0, sum23_f1, sum23_f2;
    double sum45_f0, sum45_f1, sum45_f2;
    double sum67_f0, sum67_f1, sum67_f2;
    double sum89_f0, sum89_f1, sum89_f2;

	double sum0123_f0, sum0123_f1, sum0123_f2;
	double sum4567_f0, sum4567_f1, sum4567_f2;
	double  sum0_7_f0,  sum0_7_f1,  sum0_7_f2;
	double sum_fin_f0, sum_fin_f1, sum_fin_f2;


	int row_f0;
	int col_f0;
	int row_f1;
	int col_f1;
	int row_f2;
	int col_f2;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		int cur_label = labels[shuffle_index[base+b]];

		softmax_out_0 = (softmax_out->data)[ind_softmax_out(b, 0)];
		softmax_out_1 = (softmax_out->data)[ind_softmax_out(b, 1)];
		softmax_out_2 = (softmax_out->data)[ind_softmax_out(b, 2)];
		softmax_out_3 = (softmax_out->data)[ind_softmax_out(b, 3)];
		softmax_out_4 = (softmax_out->data)[ind_softmax_out(b, 4)];
		softmax_out_5 = (softmax_out->data)[ind_softmax_out(b, 5)];
		softmax_out_6 = (softmax_out->data)[ind_softmax_out(b, 6)];
		softmax_out_7 = (softmax_out->data)[ind_softmax_out(b, 7)];
		softmax_out_8 = (softmax_out->data)[ind_softmax_out(b, 8)];
		softmax_out_9 = (softmax_out->data)[ind_softmax_out(b, 9)];


		for (int r = 0; r < N_ROWS_POOL; ++r)
		{
			for (int c = 0; c < N_COLS_POOL; ++c)
			{
				INCREMENT_FLOPS(70)
				// Unrolling the inner loop on digits completely

				delta0 = softmax_out_0 - (cur_label == 0);
				delta1 = softmax_out_1 - (cur_label == 1);
				delta2 = softmax_out_2 - (cur_label == 2);
				delta3 = softmax_out_3 - (cur_label == 3);
				delta4 = softmax_out_4 - (cur_label == 4);
				delta5 = softmax_out_5 - (cur_label == 5);
				delta6 = softmax_out_6 - (cur_label == 6);
				delta7 = softmax_out_7 - (cur_label == 7);
				delta8 = softmax_out_8 - (cur_label == 8);
				delta9 = softmax_out_9 - (cur_label == 9);

				w0_f0 = (fully_con_w->data)[ind_fully_con_w(0, 0, r, c)];
				w1_f0 = (fully_con_w->data)[ind_fully_con_w(1, 0, r, c)];
				w2_f0 = (fully_con_w->data)[ind_fully_con_w(2, 0, r, c)];
				w3_f0 = (fully_con_w->data)[ind_fully_con_w(3, 0, r, c)];
				w4_f0 = (fully_con_w->data)[ind_fully_con_w(4, 0, r, c)];
				w5_f0 = (fully_con_w->data)[ind_fully_con_w(5, 0, r, c)];
				w6_f0 = (fully_con_w->data)[ind_fully_con_w(6, 0, r, c)];
				w7_f0 = (fully_con_w->data)[ind_fully_con_w(7, 0, r, c)];
				w8_f0 = (fully_con_w->data)[ind_fully_con_w(8, 0, r, c)];
				w9_f0 = (fully_con_w->data)[ind_fully_con_w(9, 0, r, c)];

				w0_f1 = (fully_con_w->data)[ind_fully_con_w(0, 1, r, c)];
				w1_f1 = (fully_con_w->data)[ind_fully_con_w(1, 1, r, c)];
				w2_f1 = (fully_con_w->data)[ind_fully_con_w(2, 1, r, c)];
				w3_f1 = (fully_con_w->data)[ind_fully_con_w(3, 1, r, c)];
				w4_f1 = (fully_con_w->data)[ind_fully_con_w(4, 1, r, c)];
				w5_f1 = (fully_con_w->data)[ind_fully_con_w(5, 1, r, c)];
				w6_f1 = (fully_con_w->data)[ind_fully_con_w(6, 1, r, c)];
				w7_f1 = (fully_con_w->data)[ind_fully_con_w(7, 1, r, c)];
				w8_f1 = (fully_con_w->data)[ind_fully_con_w(8, 1, r, c)];
				w9_f1 = (fully_con_w->data)[ind_fully_con_w(9, 1, r, c)];

				w0_f2 = (fully_con_w->data)[ind_fully_con_w(0, 2, r, c)];
				w1_f2 = (fully_con_w->data)[ind_fully_con_w(1, 2, r, c)];
				w2_f2 = (fully_con_w->data)[ind_fully_con_w(2, 2, r, c)];
				w3_f2 = (fully_con_w->data)[ind_fully_con_w(3, 2, r, c)];
				w4_f2 = (fully_con_w->data)[ind_fully_con_w(4, 2, r, c)];
				w5_f2 = (fully_con_w->data)[ind_fully_con_w(5, 2, r, c)];
				w6_f2 = (fully_con_w->data)[ind_fully_con_w(6, 2, r, c)];
				w7_f2 = (fully_con_w->data)[ind_fully_con_w(7, 2, r, c)];
				w8_f2 = (fully_con_w->data)[ind_fully_con_w(8, 2, r, c)];
				w9_f2 = (fully_con_w->data)[ind_fully_con_w(9, 2, r, c)];


				sum0_f0 = delta0 * w0_f0;
				sum1_f0 = delta1 * w1_f0;
				sum2_f0 = delta2 * w2_f0;
				sum3_f0 = delta3 * w3_f0;
				sum4_f0 = delta4 * w4_f0;
				sum5_f0 = delta5 * w5_f0;
				sum6_f0 = delta6 * w6_f0;
				sum7_f0 = delta7 * w7_f0;
				sum8_f0 = delta8 * w8_f0;
				sum9_f0 = delta9 * w9_f0;

				sum0_f1 = delta0 * w0_f1;
				sum1_f1 = delta1 * w1_f1;
				sum2_f1 = delta2 * w2_f1;
				sum3_f1 = delta3 * w3_f1;
				sum4_f1 = delta4 * w4_f1;
				sum5_f1 = delta5 * w5_f1;
				sum6_f1 = delta6 * w6_f1;
				sum7_f1 = delta7 * w7_f1;
				sum8_f1 = delta8 * w8_f1;
				sum9_f1 = delta9 * w9_f1;

				sum0_f2 = delta0 * w0_f2;
				sum1_f2 = delta1 * w1_f2;
				sum2_f2 = delta2 * w2_f2;
				sum3_f2 = delta3 * w3_f2;
				sum4_f2 = delta4 * w4_f2;
				sum5_f2 = delta5 * w5_f2;
				sum6_f2 = delta6 * w6_f2;
				sum7_f2 = delta7 * w7_f2;
				sum8_f2 = delta8 * w8_f2;
				sum9_f2 = delta9 * w9_f2;

				sum01_f0 = sum0_f0 + sum1_f0;
				sum23_f0 = sum2_f0 + sum3_f0;
				sum45_f0 = sum4_f0 + sum5_f0;
				sum67_f0 = sum6_f0 + sum7_f0;
				sum89_f0 = sum8_f0 + sum9_f0;

				sum0123_f0 =   sum01_f0 +   sum23_f0;
				sum4567_f0 =   sum45_f0 +   sum67_f0;
				 sum0_7_f0 = sum0123_f0 + sum4567_f0;
				sum_fin_f0 =  sum0_7_f0 +   sum89_f0;

				sum01_f1 = sum0_f1 + sum1_f1;
				sum23_f1 = sum2_f1 + sum3_f1;
				sum45_f1 = sum4_f1 + sum5_f1;
				sum67_f1 = sum6_f1 + sum7_f1;
				sum89_f1 = sum8_f1 + sum9_f1;

				sum0123_f1 =   sum01_f1 +   sum23_f1;
				sum4567_f1 =   sum45_f1 +   sum67_f1;
				 sum0_7_f1 = sum0123_f1 + sum4567_f1;
				sum_fin_f1 =  sum0_7_f1 +   sum89_f1;

				sum01_f2 = sum0_f2 + sum1_f2;
				sum23_f2 = sum2_f2 + sum3_f2;
				sum45_f2 = sum4_f2 + sum5_f2;
				sum67_f2 = sum6_f2 + sum7_f2;
				sum89_f2 = sum8_f2 + sum9_f2;

				sum0123_f2 =   sum01_f2 +   sum23_f2;
				sum4567_f2 =   sum45_f2 +   sum67_f2;
				 sum0_7_f2 = sum0123_f2 + sum4567_f2;
				sum_fin_f2 =  sum0_7_f2 +   sum89_f2;


				// bp from max_pool to conv
				row_f0 = pool_index_i[b][0][r][c];
				col_f0 = pool_index_j[b][0][r][c];

				row_f1 = pool_index_i[b][1][r][c];
				col_f1 = pool_index_j[b][1][r][c];

				row_f2 = pool_index_i[b][2][r][c];
				col_f2 = pool_index_j[b][2][r][c];

				if (    conv_t->data[ind_conv_out(b, 0, row_f0, col_f0)] > 0.0)
				{
					(del_conv->data)[ind_conv_out(b, 0, row_f0, col_f0)] = sum_fin_f0;
				}

				if (    conv_t->data[ind_conv_out(b, 1, row_f1, col_f1)] > 0.0)
				{
					(del_conv->data)[ind_conv_out(b, 1, row_f1, col_f1)] = sum_fin_f1;
				}

				if (    conv_t->data[ind_conv_out(b, 2, row_f2, col_f2)] > 0.0)
				{
					(del_conv->data)[ind_conv_out(b, 2, row_f2, col_f2)] = sum_fin_f2;
				}
			}
		}
	}
}*/

/*// Vectorized: inner loops on digits and filters unrolled completely
void bp_softmax_to_conv(tensor* del_conv, tensor* softmax_out, tensor* conv_t, int* labels, int base, tensor* fully_con_w,
	int shuffle_index[], int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double delta0, delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta9;

	double softmax_out_0;
	double softmax_out_1;
	double softmax_out_2;
	double softmax_out_3;
	double softmax_out_4;
	double softmax_out_5;
	double softmax_out_6;
	double softmax_out_7;
	double softmax_out_8;
	double softmax_out_9;

	double w0_f0, w0_f1, w0_f2;
    double w1_f0, w1_f1, w1_f2;
    double w2_f0, w2_f1, w2_f2;
    double w3_f0, w3_f1, w3_f2;
    double w4_f0, w4_f1, w4_f2;
    double w5_f0, w5_f1, w5_f2;
    double w6_f0, w6_f1, w6_f2;
    double w7_f0, w7_f1, w7_f2;
    double w8_f0, w8_f1, w8_f2;
    double w9_f0, w9_f1, w9_f2;

	double sum0_f0, sum0_f1, sum0_f2;
	double sum1_f0, sum1_f1, sum1_f2;
	double sum2_f0, sum2_f1, sum2_f2;
	double sum3_f0, sum3_f1, sum3_f2;
	double sum4_f0, sum4_f1, sum4_f2;
	double sum5_f0, sum5_f1, sum5_f2;
	double sum6_f0, sum6_f1, sum6_f2;
	double sum7_f0, sum7_f1, sum7_f2;
	double sum8_f0, sum8_f1, sum8_f2;
	double sum9_f0, sum9_f1, sum9_f2;

    double sum01_f0, sum01_f1, sum01_f2;
    double sum23_f0, sum23_f1, sum23_f2;
    double sum45_f0, sum45_f1, sum45_f2;
    double sum67_f0, sum67_f1, sum67_f2;
    double sum89_f0, sum89_f1, sum89_f2;

	double sum0123_f0, sum0123_f1, sum0123_f2;
	double sum4567_f0, sum4567_f1, sum4567_f2;
	double  sum0_7_f0,  sum0_7_f1,  sum0_7_f2;
	double sum_fin_f0, sum_fin_f1, sum_fin_f2;

	__m256d softmax_out_1_p;
	__m256d softmax_out_2_p;
	__m256d softmax_out_3_p;

	__m256d expected_out_1_p;
	__m256d expected_out_2_p;
	__m256d expected_out_3_p;

	__m256d delta_1_p;
	__m256d delta_2_p;
	__m256d delta_3_p;

	__m256d w_1_f0_p, w_1_f1_p, w_1_f2_p;
	__m256d w_2_f0_p, w_2_f1_p, w_2_f2_p;
	__m256d w_3_f0_p, w_3_f1_p, w_3_f2_p;

	__m256d sum_1_f0_p, sum_1_f1_p, sum_1_f2_p;
	__m256d sum_2_f0_p, sum_2_f1_p, sum_2_f2_p;
	__m256d sum_3_f0_p, sum_3_f1_p, sum_3_f2_p;

	int row_f0;
	int col_f0;
	int row_f1;
	int col_f1;
	int row_f2;
	int col_f2;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		int cur_label = labels[shuffle_index[base+b]];

		//softmax_out_0 = (softmax_out->data)[ind_softmax_out(b, 0)];
		//softmax_out_1 = (softmax_out->data)[ind_softmax_out(b, 1)];
		//softmax_out_2 = (softmax_out->data)[ind_softmax_out(b, 2)];
		//softmax_out_3 = (softmax_out->data)[ind_softmax_out(b, 3)];
		//softmax_out_4 = (softmax_out->data)[ind_softmax_out(b, 4)];
		//softmax_out_5 = (softmax_out->data)[ind_softmax_out(b, 5)];
		//softmax_out_6 = (softmax_out->data)[ind_softmax_out(b, 6)];
		//softmax_out_7 = (softmax_out->data)[ind_softmax_out(b, 7)];
		//softmax_out_8 = (softmax_out->data)[ind_softmax_out(b, 8)];
		//softmax_out_9 = (softmax_out->data)[ind_softmax_out(b, 9)];

		softmax_out_1_p = _mm256_loadu_pd( (softmax_out->data) + ind_softmax_out(b, 0) );
		softmax_out_2_p = _mm256_loadu_pd( (softmax_out->data) + ind_softmax_out(b, 4) );
		softmax_out_3_p = _mm256_loadu_pd( (softmax_out->data) + ind_softmax_out(b, 8) );

		expected_out_1_p = _mm256_set_pd( (cur_label == 3), (cur_label == 2), (cur_label == 1), (cur_label == 0));
		expected_out_2_p = _mm256_set_pd( (cur_label == 7), (cur_label == 6), (cur_label == 5), (cur_label == 4));
		expected_out_3_p = _mm256_set_pd( 0               , 0               , (cur_label == 9), (cur_label == 8));


		for (int r = 0; r < N_ROWS_POOL; ++r)
		{
			for (int c = 0; c < N_COLS_POOL; ++c)
			{
				INCREMENT_FLOPS(70)
				// Unrolling the inner loop on digits completely

				//delta0 = softmax_out_0 - (cur_label == 0);
				//delta1 = softmax_out_1 - (cur_label == 1);
				//delta2 = softmax_out_2 - (cur_label == 2);
				//delta3 = softmax_out_3 - (cur_label == 3);
				//delta4 = softmax_out_4 - (cur_label == 4);
				//delta5 = softmax_out_5 - (cur_label == 5);
				//delta6 = softmax_out_6 - (cur_label == 6);
				//delta7 = softmax_out_7 - (cur_label == 7);
				//delta8 = softmax_out_8 - (cur_label == 8);
				//delta9 = softmax_out_9 - (cur_label == 9);

				delta_1_p = _mm256_sub_pd( softmax_out_1_p, expected_out_1_p);
				delta_2_p = _mm256_sub_pd( softmax_out_2_p, expected_out_2_p);
				delta_3_p = _mm256_sub_pd( softmax_out_3_p, expected_out_3_p);

				w0_f0 = (fully_con_w->data)[ind_fully_con_w(0, 0, r, c)];
				w1_f0 = (fully_con_w->data)[ind_fully_con_w(1, 0, r, c)];
				w2_f0 = (fully_con_w->data)[ind_fully_con_w(2, 0, r, c)];
				w3_f0 = (fully_con_w->data)[ind_fully_con_w(3, 0, r, c)];
				w4_f0 = (fully_con_w->data)[ind_fully_con_w(4, 0, r, c)];
				w5_f0 = (fully_con_w->data)[ind_fully_con_w(5, 0, r, c)];
				w6_f0 = (fully_con_w->data)[ind_fully_con_w(6, 0, r, c)];
				w7_f0 = (fully_con_w->data)[ind_fully_con_w(7, 0, r, c)];
				w8_f0 = (fully_con_w->data)[ind_fully_con_w(8, 0, r, c)];
				w9_f0 = (fully_con_w->data)[ind_fully_con_w(9, 0, r, c)];

				w_1_f0_p = _mm256_set_pd(w3_f0, w2_f0, w1_f0, w0_f0);
				w_2_f0_p = _mm256_set_pd(w7_f0, w6_f0, w5_f0, w4_f0);
				w_3_f0_p = _mm256_set_pd(0    , 0    , w9_f0, w8_f0);

				w0_f1 = (fully_con_w->data)[ind_fully_con_w(0, 1, r, c)];
				w1_f1 = (fully_con_w->data)[ind_fully_con_w(1, 1, r, c)];
				w2_f1 = (fully_con_w->data)[ind_fully_con_w(2, 1, r, c)];
				w3_f1 = (fully_con_w->data)[ind_fully_con_w(3, 1, r, c)];
				w4_f1 = (fully_con_w->data)[ind_fully_con_w(4, 1, r, c)];
				w5_f1 = (fully_con_w->data)[ind_fully_con_w(5, 1, r, c)];
				w6_f1 = (fully_con_w->data)[ind_fully_con_w(6, 1, r, c)];
				w7_f1 = (fully_con_w->data)[ind_fully_con_w(7, 1, r, c)];
				w8_f1 = (fully_con_w->data)[ind_fully_con_w(8, 1, r, c)];
				w9_f1 = (fully_con_w->data)[ind_fully_con_w(9, 1, r, c)];

				w_1_f1_p = _mm256_set_pd(w3_f1, w2_f1, w1_f1, w0_f1);
				w_2_f1_p = _mm256_set_pd(w7_f1, w6_f1, w5_f1, w4_f1);
				w_3_f1_p = _mm256_set_pd(0    , 0    , w9_f1, w8_f1);

				w0_f2 = (fully_con_w->data)[ind_fully_con_w(0, 2, r, c)];
				w1_f2 = (fully_con_w->data)[ind_fully_con_w(1, 2, r, c)];
				w2_f2 = (fully_con_w->data)[ind_fully_con_w(2, 2, r, c)];
				w3_f2 = (fully_con_w->data)[ind_fully_con_w(3, 2, r, c)];
				w4_f2 = (fully_con_w->data)[ind_fully_con_w(4, 2, r, c)];
				w5_f2 = (fully_con_w->data)[ind_fully_con_w(5, 2, r, c)];
				w6_f2 = (fully_con_w->data)[ind_fully_con_w(6, 2, r, c)];
				w7_f2 = (fully_con_w->data)[ind_fully_con_w(7, 2, r, c)];
				w8_f2 = (fully_con_w->data)[ind_fully_con_w(8, 2, r, c)];
				w9_f2 = (fully_con_w->data)[ind_fully_con_w(9, 2, r, c)];

				w_1_f2_p = _mm256_set_pd(w3_f2, w2_f2, w1_f2, w0_f2);
				w_2_f2_p = _mm256_set_pd(w7_f2, w6_f2, w5_f2, w4_f2);
				w_3_f2_p = _mm256_set_pd(0    , 0    , w9_f2, w8_f2);


				//sum0_f0 = delta0 * w0_f0;
				//sum1_f0 = delta1 * w1_f0;
				//sum2_f0 = delta2 * w2_f0;
				//sum3_f0 = delta3 * w3_f0;
				//sum4_f0 = delta4 * w4_f0;
				//sum5_f0 = delta5 * w5_f0;
				//sum6_f0 = delta6 * w6_f0;
				//sum7_f0 = delta7 * w7_f0;
				//sum8_f0 = delta8 * w8_f0;
				//sum9_f0 = delta9 * w9_f0;

				sum_1_f0_p = _mm256_mul_pd(delta_1_p, w_1_f0_p);
				sum_2_f0_p = _mm256_mul_pd(delta_2_p, w_2_f0_p);
				sum_3_f0_p = _mm256_mul_pd(delta_3_p, w_3_f0_p);

				//sum0_f1 = delta0 * w0_f1;
				//sum1_f1 = delta1 * w1_f1;
				//sum2_f1 = delta2 * w2_f1;
				//sum3_f1 = delta3 * w3_f1;
				//sum4_f1 = delta4 * w4_f1;
				//sum5_f1 = delta5 * w5_f1;
				//sum6_f1 = delta6 * w6_f1;
				//sum7_f1 = delta7 * w7_f1;
				//sum8_f1 = delta8 * w8_f1;
				//sum9_f1 = delta9 * w9_f1;

				sum_1_f1_p = _mm256_mul_pd(delta_1_p, w_1_f1_p);
				sum_2_f1_p = _mm256_mul_pd(delta_2_p, w_2_f1_p);
				sum_3_f1_p = _mm256_mul_pd(delta_3_p, w_3_f1_p);

				//sum0_f2 = delta0 * w0_f2;
				//sum1_f2 = delta1 * w1_f2;
				//sum2_f2 = delta2 * w2_f2;
				//sum3_f2 = delta3 * w3_f2;
				//sum4_f2 = delta4 * w4_f2;
				//sum5_f2 = delta5 * w5_f2;
				//sum6_f2 = delta6 * w6_f2;
				//sum7_f2 = delta7 * w7_f2;
				//sum8_f2 = delta8 * w8_f2;
				//sum9_f2 = delta9 * w9_f2;

				sum_1_f2_p = _mm256_mul_pd(delta_1_p, w_1_f2_p);
				sum_2_f2_p = _mm256_mul_pd(delta_2_p, w_2_f2_p);
				sum_3_f2_p = _mm256_mul_pd(delta_3_p, w_3_f2_p);

				//-----------------------------------------------------------------------

				sum01_f0 = sum_1_f0_p[0] + sum_1_f0_p[1];
				sum23_f0 = sum_1_f0_p[2] + sum_1_f0_p[3];
				sum45_f0 = sum_2_f0_p[0] + sum_2_f0_p[1];
				sum67_f0 = sum_2_f0_p[2] + sum_2_f0_p[3];
				sum89_f0 = sum_3_f0_p[0] + sum_3_f0_p[1];

				sum0123_f0 =   sum01_f0 +   sum23_f0;
				sum4567_f0 =   sum45_f0 +   sum67_f0;
				 sum0_7_f0 = sum0123_f0 + sum4567_f0;
				sum_fin_f0 =  sum0_7_f0 +   sum89_f0;

				sum01_f1 = sum_1_f1_p[0] + sum_1_f1_p[1];
				sum23_f1 = sum_1_f1_p[2] + sum_1_f1_p[3];
				sum45_f1 = sum_2_f1_p[0] + sum_2_f1_p[1];
				sum67_f1 = sum_2_f1_p[2] + sum_2_f1_p[3];
				sum89_f1 = sum_3_f1_p[0] + sum_3_f1_p[1];

				sum0123_f1 =   sum01_f1 +   sum23_f1;
				sum4567_f1 =   sum45_f1 +   sum67_f1;
				 sum0_7_f1 = sum0123_f1 + sum4567_f1;
				sum_fin_f1 =  sum0_7_f1 +   sum89_f1;

				sum01_f2 = sum_1_f2_p[0] + sum_1_f2_p[1];
				sum23_f2 = sum_1_f2_p[2] + sum_1_f2_p[3];
				sum45_f2 = sum_2_f2_p[0] + sum_2_f2_p[1];
				sum67_f2 = sum_2_f2_p[2] + sum_2_f2_p[3];
				sum89_f2 = sum_3_f2_p[0] + sum_3_f2_p[1];

				sum0123_f2 =   sum01_f2 +   sum23_f2;
				sum4567_f2 =   sum45_f2 +   sum67_f2;
				 sum0_7_f2 = sum0123_f2 + sum4567_f2;
				sum_fin_f2 =  sum0_7_f2 +   sum89_f2;


				// bp from max_pool to conv
				row_f0 = pool_index_i[b][0][r][c];
				col_f0 = pool_index_j[b][0][r][c];

				row_f1 = pool_index_i[b][1][r][c];
				col_f1 = pool_index_j[b][1][r][c];

				row_f2 = pool_index_i[b][2][r][c];
				col_f2 = pool_index_j[b][2][r][c];

				if (    conv_t->data[ind_conv_out(b, 0, row_f0, col_f0)] > 0.0)
				{
					(del_conv->data)[ind_conv_out(b, 0, row_f0, col_f0)] = sum_fin_f0;
				}

				if (    conv_t->data[ind_conv_out(b, 1, row_f1, col_f1)] > 0.0)
				{
					(del_conv->data)[ind_conv_out(b, 1, row_f1, col_f1)] = sum_fin_f1;
				}

				if (    conv_t->data[ind_conv_out(b, 2, row_f2, col_f2)] > 0.0)
				{
					(del_conv->data)[ind_conv_out(b, 2, row_f2, col_f2)] = sum_fin_f2;
				}
			}
		}
	}
}*/

// Vectorized: pool cols only; best perf
void bp_softmax_to_conv(tensor* del_conv, tensor* softmax_out, tensor* conv_t, int* labels, int base, tensor* fully_con_w,
	int shuffle_index[], int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double delta;
	double out;
	double w;
	double sum;

	int row;
	int col;

	__m256d delta_p;
	__m256d     w_p;
	__m256d   sum_p;
	__m256d   conv_val_p;

	int row_c0, row_c1, row_c2, row_c3;
	int col_c0, col_c1, col_c2, col_c3;

	int c, d;

	int cur_label;

	__m256d zeroes_p = _mm256_set1_pd(0.0);
	__m256d   ones_p = _mm256_set1_pd(1);

	__m256d if_result_p, vmask, vmask1, d_p;

	__m256d delta_p_d0;
	__m256d delta_p_d1;
	__m256d delta_p_d2;
	__m256d delta_p_d3;

	__m256d w_p_d0;
	__m256d w_p_d1;
	__m256d w_p_d2;
	__m256d w_p_d3;

	__m256d sum_p_d0;
	__m256d sum_p_d1;
	__m256d sum_p_d2;
	__m256d sum_p_d3;

	__m256d sum_p_temp1;
	__m256d sum_p_temp2;
	__m256d sum_p_temp3;

	__m256d cur_label_p;
	__m256d sum_p_rem;
	__m256d out_p;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		cur_label   = labels[shuffle_index[base+b]];
		cur_label_p = _mm256_set1_pd(cur_label);

		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (c = 0; c+3 < N_COLS_POOL; c=c+4)
				{
					sum_p_rem = _mm256_set1_pd(0.0);
					sum_p_d0  = _mm256_set1_pd(0.0);
					sum_p_d1  = _mm256_set1_pd(0.0);
					sum_p_d2  = _mm256_set1_pd(0.0);
					sum_p_d3  = _mm256_set1_pd(0.0);

					for (d = 0; d+3 < N_DIGS; d=d+4)
					{

						INCREMENT_FLOPS(40)

						out_p = _mm256_loadu_pd( (softmax_out->data) + ind_softmax_out(b, d) );
						d_p   = _mm256_set_pd(d+3, d+2, d+1, d);

						vmask1 = _mm256_cmp_pd(cur_label_p, d_p, 0x10); // 0x10 => EQ.

						delta_p  = _mm256_sub_pd( out_p, _mm256_and_pd(ones_p, vmask1) );

						delta_p_d0 = _mm256_set1_pd(delta_p[0]);
						delta_p_d1 = _mm256_set1_pd(delta_p[1]);
						delta_p_d2 = _mm256_set1_pd(delta_p[2]);
						delta_p_d3 = _mm256_set1_pd(delta_p[3]);

						w_p_d0 = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d  , f, r, c) );
						w_p_d1 = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d+1, f, r, c) );
						w_p_d2 = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d+2, f, r, c) );
						w_p_d3 = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d+3, f, r, c) );

						sum_p_d0 = _mm256_add_pd( sum_p_d0, _mm256_mul_pd(delta_p_d0, w_p_d0) );
						sum_p_d1 = _mm256_add_pd( sum_p_d1, _mm256_mul_pd(delta_p_d1, w_p_d1) );
						sum_p_d2 = _mm256_add_pd( sum_p_d2, _mm256_mul_pd(delta_p_d2, w_p_d2) );
						sum_p_d3 = _mm256_add_pd( sum_p_d3, _mm256_mul_pd(delta_p_d3, w_p_d3) );
					}

					for (; d < N_DIGS; ++d)
					{

						INCREMENT_FLOPS(9)

						out = (softmax_out->data)[ind_softmax_out(b, d)];

						delta = out - (cur_label == d);

						delta_p = _mm256_set1_pd(delta);

						w_p = _mm256_loadu_pd( (fully_con_w->data) + ind_fully_con_w(d, f, r, c) );

						sum_p_rem = _mm256_add_pd( sum_p_rem, _mm256_mul_pd(delta_p, w_p) );
					}

					INCREMENT_FLOPS(20)

					sum_p_temp1 = _mm256_add_pd(sum_p_d0, sum_p_d1);
					sum_p_temp2 = _mm256_add_pd(sum_p_d2, sum_p_d3);
					sum_p_temp3 = _mm256_add_pd(sum_p_temp1, sum_p_temp2);
					sum_p       = _mm256_add_pd(sum_p_temp3, sum_p_rem);

					// bp from max_pool to conv
					row_c0 = pool_index_i[b][f][r][c  ];
					col_c0 = pool_index_j[b][f][r][c  ];

					row_c1 = pool_index_i[b][f][r][c+1];
					col_c1 = pool_index_j[b][f][r][c+1];

					row_c2 = pool_index_i[b][f][r][c+2];
					col_c2 = pool_index_j[b][f][r][c+2];

					row_c3 = pool_index_i[b][f][r][c+3];
					col_c3 = pool_index_j[b][f][r][c+3];

					conv_val_p = _mm256_set_pd( conv_t->data[ind_conv_out(b, f, row_c0, col_c0)],
												conv_t->data[ind_conv_out(b, f, row_c1, col_c1)],
												conv_t->data[ind_conv_out(b, f, row_c2, col_c2)],
												conv_t->data[ind_conv_out(b, f, row_c3, col_c3)]
											);

					vmask = _mm256_cmp_pd(conv_val_p, zeroes_p, 0x0e); // 0x0e => GT.

					if_result_p = _mm256_and_pd( vmask, sum_p);

					(del_conv->data)[ind_conv_out(b, f, row_c0, col_c0)] = if_result_p[0];
					(del_conv->data)[ind_conv_out(b, f, row_c1, col_c1)] = if_result_p[1];
					(del_conv->data)[ind_conv_out(b, f, row_c2, col_c2)] = if_result_p[2];
					(del_conv->data)[ind_conv_out(b, f, row_c3, col_c3)] = if_result_p[3];
				}

				for (; c < N_COLS_POOL; ++c)
				{
					INCREMENT_FLOPS(3)

					sum = 0.0;

					for (int d = 0; d < N_DIGS; ++d)
					{

						out = (softmax_out->data)[ind_softmax_out(b, d)];

						delta = out - (cur_label == d);

						w = (fully_con_w->data)[ind_fully_con_w(d, f, r, c)];

						sum += delta * w;
					}

					// bp from max_pool to conv
					row = pool_index_i[b][f][r][c];
					col = pool_index_j[b][f][r][c];

					INCREMENT_FLOPS(1)

					if (    conv_t->data[ind_conv_out(b, f, row, col)] > 0.0)
					{
						(del_conv->data)[ind_conv_out(b, f, row, col)] = sum;
					}
				}
			}
		}
	}
}

void update_conv_weights(tensor* fil_w, tensor* del_conv, tensor* conv_t, tensor* input_images, int base, int shuffle_index[])
{
	int multiplier = (LEARN_RATE/BATCH_SIZE);
	for (int f = 0; f < NUM_FILS; ++f)
	{
		for (int r = 0; r < FIL_ROWS; ++r)
		{
			for (int c = 0; c < FIL_COLS; ++c)
			{
		    	double delta_w = 0.0;

		    	for (int b = 0; b < BATCH_SIZE; ++b)
		    	{
		    		int cur_image = shuffle_index[b+base];
		    		for (int i = 0; i < N_ROWS_CONV; ++i)
		    		{
		    			for (int j = 0; j < N_COLS_CONV; ++j)
		    			{
		    				if( (conv_t->data)[offset(conv_t, b, j , i, f)] > 0.0 )
		    				{
		    					delta_w += (del_conv->data)[offset(del_conv, b, j , i, f)]*
		    									(input_images->data)[offset(input_images, cur_image, j+c, i+r, 0)];
		    				}
		    			}
		    		}
		    	}

		    	(fil_w->data)[offset(fil_w, f, r, c, 0)] -= multiplier*delta_w;
			}
		}
	}
}

// Vectorized: conv cols, filter cols; number of filters
/*void bin_update_conv_weights(tensor* fil_w, tensor* fil_bin_w, double alphas[NUM_FILS], tensor* del_conv, tensor* conv_t,
								tensor* input_images, int base, int shuffle_index[])
{
	INCREMENT_FLOPS(1)

	double recip_n    = 1.0/(FIL_ROWS * FIL_COLS);
	__m256d recip_n_p = _mm256_set1_pd(recip_n);

	__m256d multiplier_p = _mm256_set1_pd(MULTIPLIER);

	__m256d zeroes_p = _mm256_set1_pd( 0.0);
	__m256d ones_p   = _mm256_set1_pd( 1.0);
	__m256d mones_p  = _mm256_set1_pd(-1.0);

	__m256d delta_w1_p_c0_f0, delta_w1_p_c0_f1, delta_w1_p_c0_f2;
	__m256d delta_w1_p_c1_f0, delta_w1_p_c1_f1, delta_w1_p_c1_f2;
	__m256d delta_w1_p_c2_f0, delta_w1_p_c2_f1, delta_w1_p_c2_f2;
	__m256d delta_w1_p_c3_f0, delta_w1_p_c3_f1, delta_w1_p_c3_f2;

	__m256d delta_w1_p_rem_f0;
	__m256d delta_w1_p_rem_f1;
	__m256d delta_w1_p_rem_f2;

	int cur_image1;

	__m256d conv_val1_p_f0;
	__m256d conv_val1_p_f1;
	__m256d conv_val1_p_f2;

	__m256d del_conv_val1_p_f0;
	__m256d del_conv_val1_p_f1;
	__m256d del_conv_val1_p_f2;

	__m256d input_pixel1_p_c0;
	__m256d input_pixel1_p_c1;
	__m256d input_pixel1_p_c2;
	__m256d input_pixel1_p_c3;


	__m256d vmask1_f0, vmask1_f1, vmask1_f2;
	__m256d and1_p_f0, and1_p_f1, and1_p_f2;

	__m256d mul1_p_c0_f0, mul1_p_c0_f1, mul1_p_c0_f2;
	__m256d mul1_p_c1_f0, mul1_p_c1_f1, mul1_p_c1_f2;
	__m256d mul1_p_c2_f0, mul1_p_c2_f1, mul1_p_c2_f2;
	__m256d mul1_p_c3_f0, mul1_p_c3_f1, mul1_p_c3_f2;


	double conv_val1_rem_f0, conv_val1_rem_f1, conv_val1_rem_f2;

	__m256d del_conv_val1_p_rem_f0, del_conv_val1_p_rem_f1, del_conv_val1_p_rem_f2;

	__m256d input_pixel1_p_rem;

	double delta_w1_c0_f0, delta_w1_c0_f1, delta_w1_c0_f2;
	double delta_w1_c1_f0, delta_w1_c1_f1, delta_w1_c1_f2;
	double delta_w1_c2_f0, delta_w1_c2_f1, delta_w1_c2_f2;
	double delta_w1_c3_f0, delta_w1_c3_f1, delta_w1_c3_f2;

	__m256d delta_ws1_p_f0, delta_ws1_p_f1, delta_ws1_p_f2;


	__m256d weights1_p_f0, weights1_p_f1, weights1_p_f2;

	__m256d vmask_ones1_f0, vmask_ones1_f1, vmask_ones1_f2;

	__m256d vmask_mones1_f0, vmask_mones1_f1, vmask_mones1_f2;

	__m256d vmask_final1_f0, vmask_final1_f1, vmask_final1_f2;

	__m256d alphas_p_f0, alphas_p_f1, alphas_p_f2;

	__m256d if_result1_p_f0, if_result1_p_f1, if_result1_p_f2;

	__m256d delta_w_if1_p_f0, delta_w_if1_p_f1, delta_w_if1_p_f2;

	__m256d delta_w_else1_p_f0, delta_w_else1_p_f1, delta_w_else1_p_f2;

    __m256d delta_w2_p_f0;
    __m256d delta_w2_p_f1;
    __m256d delta_w2_p_f2;

    double delta_w2_rem_f0;
    double delta_w2_rem_f1;
    double delta_w2_rem_f2;

    int cur_image2;

	__m256d conv_val2_p_f0;
	__m256d conv_val2_p_f1;
	__m256d conv_val2_p_f2;

	__m256d del_conv_val2_p_f0;
	__m256d del_conv_val2_p_f1;
	__m256d del_conv_val2_p_f2;

	__m256d input_pixel2_p;

	__m256d vmask2_f0, vmask2_f1, vmask2_f2;

	__m256d and2_p_f0, and2_p_f1, and2_p_f2;

	__m256d mul2_p_f0, mul2_p_f1, mul2_p_f2;

	double conv_val2_rem_f0, conv_val2_rem_f1, conv_val2_rem_f2;

	double del_conv_val2_rem_f0, del_conv_val2_rem_f1, del_conv_val2_rem_f2;

	double input_pixel2_rem;

	double delta_w2_f0, delta_w2_f1, delta_w2_f2;

	double weight2_f0, weight2_f1, weight2_f2;

	__m256d delta_w3_p_c0;
	__m256d delta_w3_p_c1;
	__m256d delta_w3_p_c2;
	__m256d delta_w3_p_c3;

	__m256d delta_w3_p_rem;

	int cur_image3;

	__m256d conv_val3_p;

	__m256d del_conv_val3_p;

	__m256d input_pixel3_p_c0;
	__m256d input_pixel3_p_c1;
	__m256d input_pixel3_p_c2;
	__m256d input_pixel3_p_c3;

	__m256d vmask3;

	__m256d and3_p;

	__m256d mul3_p_c0;
	__m256d mul3_p_c1;
	__m256d mul3_p_c2;
	__m256d mul3_p_c3;

	double conv_val3_rem;
	__m256d del_conv_val3_p_rem;
	__m256d input_pixel3_p_rem;

	double delta_w3_c0;
	double delta_w3_c1;
	double delta_w3_c2;
	double delta_w3_c3;

	__m256d delta_ws3_p;

	__m256d weights3_p;

	__m256d vmask_ones3;

	__m256d vmask_mones3;

	__m256d vmask_final3;

	__m256d if_result3_p;

	__m256d delta_w_if3_p;

	__m256d delta_w_else3_p;


	__m256d delta_w4_p;

	double delta_w4_rem;

	int cur_image4;

	__m256d conv_val4_p;
	__m256d del_conv_val4_p;
	__m256d input_pixel4_p;
	__m256d vmask4;
	__m256d and4_p;
	__m256d mul4_p;

	double conv_val4_rem;
	double del_conv_val4_rem;
	double input_pixel4_rem;

	double delta_w4;
	double weight4;

	__m256d alphas_p;

	__m256d delta_w1_1_p_f0;
	__m256d delta_w1_2_p_f0;
	__m256d perm_1_f0_p;
	__m256d perm_2_f0_p;

	__m256d delta_w1_1_p_f1;
	__m256d delta_w1_2_p_f1;
	__m256d perm_1_f1_p;
	__m256d perm_2_f1_p;

	__m256d delta_w1_1_p_f2;
	__m256d delta_w1_2_p_f2;
	__m256d perm_1_f2_p;
	__m256d perm_2_f2_p;

	__m256d delta_w3_1_p;
	__m256d delta_w3_2_p;
	__m256d perm_1_p;
	__m256d perm_2_p;

	__m256d prev_c1_p;
	__m256d prev_c2_p;
	__m256d prev_c3_p;

	int f, c, j;

    for (f = 0; f+2 < NUM_FILS; f=f+3)
	{

		alphas_p_f0 = _mm256_set1_pd(alphas[f  ]);
		alphas_p_f1 = _mm256_set1_pd(alphas[f+1]);
		alphas_p_f2 = _mm256_set1_pd(alphas[f+2]);

		for (int r = 0; r < FIL_ROWS; ++r)
		{

			for (c = 0; c+3 < FIL_COLS; c=c+4)
			{

			    delta_w1_p_c0_f0  = _mm256_set1_pd(0.0);
			    delta_w1_p_c1_f0  = _mm256_set1_pd(0.0);
			    delta_w1_p_c2_f0  = _mm256_set1_pd(0.0);
			    delta_w1_p_c3_f0  = _mm256_set1_pd(0.0);

			    delta_w1_p_c0_f1  = _mm256_set1_pd(0.0);
			    delta_w1_p_c1_f1  = _mm256_set1_pd(0.0);
			    delta_w1_p_c2_f1  = _mm256_set1_pd(0.0);
			    delta_w1_p_c3_f1  = _mm256_set1_pd(0.0);

			    delta_w1_p_c0_f2  = _mm256_set1_pd(0.0);
			    delta_w1_p_c1_f2  = _mm256_set1_pd(0.0);
			    delta_w1_p_c2_f2  = _mm256_set1_pd(0.0);
			    delta_w1_p_c3_f2  = _mm256_set1_pd(0.0);

			    delta_w1_p_rem_f0 = _mm256_set1_pd(0.0);
			    delta_w1_p_rem_f1 = _mm256_set1_pd(0.0);
			    delta_w1_p_rem_f2 = _mm256_set1_pd(0.0);

			    for (int b = 0; b < BATCH_SIZE; ++b)
			    {

			    	cur_image1 = shuffle_index[b+base];

	    			for (int i = 0; i < N_ROWS_CONV; ++i)
	    			{
	    				prev_c1_p = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image1, i+r, c  ) );
	    				prev_c2_p = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image1, i+r, c+1) );
	    				prev_c3_p = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image1, i+r, c+2) );

	    				for (j = 0; j+3 < N_COLS_CONV; j=j+4)
	    				{

	    					conv_val1_p_f0     = _mm256_loadu_pd( (conv_t->data)       + ind_conv_out(b, f  , i, j) );
	    					conv_val1_p_f1     = _mm256_loadu_pd( (conv_t->data)       + ind_conv_out(b, f+1, i, j) );
	    					conv_val1_p_f2     = _mm256_loadu_pd( (conv_t->data)       + ind_conv_out(b, f+2, i, j) );

	    					del_conv_val1_p_f0 = _mm256_loadu_pd( (del_conv->data)     + ind_conv_out(b, f  , i, j) );
	    					del_conv_val1_p_f1 = _mm256_loadu_pd( (del_conv->data)     + ind_conv_out(b, f+1, i, j) );
	    					del_conv_val1_p_f2 = _mm256_loadu_pd( (del_conv->data)     + ind_conv_out(b, f+2, i, j) );

	    					input_pixel1_p_c0  = prev_c1_p;
	    					input_pixel1_p_c1  = prev_c2_p;
	    					input_pixel1_p_c2  = prev_c3_p;
	    					input_pixel1_p_c3  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image1, i+r, j+c+3) );


	    					INCREMENT_FLOPS(108)

	    					// ------------------------------------derivative of ReLU-------------------------------------------------------

	    					//------------------------------------------filter 0------------------------------------------------------------
	    					vmask1_f0 = _mm256_cmp_pd(conv_val1_p_f0, zeroes_p, 0x0e); // 0x0e => GT.
	    					and1_p_f0 = _mm256_and_pd(del_conv_val1_p_f0, vmask1_f0); // zero out the del_convs for which convs are less than equal to zero

	    					mul1_p_c0_f0     = _mm256_mul_pd(and1_p_f0, input_pixel1_p_c0);
	    					mul1_p_c1_f0     = _mm256_mul_pd(and1_p_f0, input_pixel1_p_c1);
	    					mul1_p_c2_f0     = _mm256_mul_pd(and1_p_f0, input_pixel1_p_c2);
	    					mul1_p_c3_f0     = _mm256_mul_pd(and1_p_f0, input_pixel1_p_c3);

	    					delta_w1_p_c0_f0 = _mm256_add_pd(delta_w1_p_c0_f0, mul1_p_c0_f0);
	    					delta_w1_p_c1_f0 = _mm256_add_pd(delta_w1_p_c1_f0, mul1_p_c1_f0);
	    					delta_w1_p_c2_f0 = _mm256_add_pd(delta_w1_p_c2_f0, mul1_p_c2_f0);
	    					delta_w1_p_c3_f0 = _mm256_add_pd(delta_w1_p_c3_f0, mul1_p_c3_f0);

	    					//------------------------------------------filter 1------------------------------------------------------------
	    					vmask1_f1 = _mm256_cmp_pd(conv_val1_p_f1, zeroes_p, 0x0e); // 0x0e => GT.
	    					and1_p_f1 = _mm256_and_pd(del_conv_val1_p_f1, vmask1_f1); // zero out the del_convs for which convs are less than equal to zero

	    					mul1_p_c0_f1     = _mm256_mul_pd(and1_p_f1, input_pixel1_p_c0);
	    					mul1_p_c1_f1     = _mm256_mul_pd(and1_p_f1, input_pixel1_p_c1);
	    					mul1_p_c2_f1     = _mm256_mul_pd(and1_p_f1, input_pixel1_p_c2);
	    					mul1_p_c3_f1     = _mm256_mul_pd(and1_p_f1, input_pixel1_p_c3);

	    					delta_w1_p_c0_f1 = _mm256_add_pd(delta_w1_p_c0_f1, mul1_p_c0_f1);
	    					delta_w1_p_c1_f1 = _mm256_add_pd(delta_w1_p_c1_f1, mul1_p_c1_f1);
	    					delta_w1_p_c2_f1 = _mm256_add_pd(delta_w1_p_c2_f1, mul1_p_c2_f1);
	    					delta_w1_p_c3_f1 = _mm256_add_pd(delta_w1_p_c3_f1, mul1_p_c3_f1);

	    					//------------------------------------------filter 0------------------------------------------------------------
	    					vmask1_f2 = _mm256_cmp_pd(conv_val1_p_f2, zeroes_p, 0x0e); // 0x0e => GT.
	    					and1_p_f2 = _mm256_and_pd(del_conv_val1_p_f2, vmask1_f2); // zero out the del_convs for which convs are less than equal to zero

	    					mul1_p_c0_f2     = _mm256_mul_pd(and1_p_f2, input_pixel1_p_c0);
	    					mul1_p_c1_f2     = _mm256_mul_pd(and1_p_f2, input_pixel1_p_c1);
	    					mul1_p_c2_f2     = _mm256_mul_pd(and1_p_f2, input_pixel1_p_c2);
	    					mul1_p_c3_f2     = _mm256_mul_pd(and1_p_f2, input_pixel1_p_c3);

	    					delta_w1_p_c0_f2 = _mm256_add_pd(delta_w1_p_c0_f2, mul1_p_c0_f2);
	    					delta_w1_p_c1_f2 = _mm256_add_pd(delta_w1_p_c1_f2, mul1_p_c1_f2);
	    					delta_w1_p_c2_f2 = _mm256_add_pd(delta_w1_p_c2_f2, mul1_p_c2_f2);
	    					delta_w1_p_c3_f2 = _mm256_add_pd(delta_w1_p_c3_f2, mul1_p_c3_f2);


	    					// store already loaded pixels
	    					prev_c1_p = input_pixel1_p_c1;
	    					prev_c2_p = input_pixel1_p_c2;
	    					prev_c3_p = input_pixel1_p_c3;
	    				}
	    			}
			    }

			    INCREMENT_FLOPS(132)

			    // horizontally sum the four vectors
				delta_w1_1_p_f0 = _mm256_hadd_pd(delta_w1_p_c0_f0, delta_w1_p_c2_f0);
				delta_w1_2_p_f0 = _mm256_hadd_pd(delta_w1_p_c1_f0, delta_w1_p_c3_f0);

				perm_1_f0_p = _mm256_permute4x64_pd(delta_w1_1_p_f0, _MM_SHUFFLE(3,1,2,0));
				perm_2_f0_p = _mm256_permute4x64_pd(delta_w1_2_p_f0, _MM_SHUFFLE(3,1,2,0));

				delta_ws1_p_f0 = _mm256_hadd_pd( perm_1_f0_p, perm_2_f0_p );



				delta_w1_1_p_f1 = _mm256_hadd_pd(delta_w1_p_c0_f1, delta_w1_p_c2_f1);
				delta_w1_2_p_f1 = _mm256_hadd_pd(delta_w1_p_c1_f1, delta_w1_p_c3_f1);

				perm_1_f1_p = _mm256_permute4x64_pd(delta_w1_1_p_f1, _MM_SHUFFLE(3,1,2,0));
				perm_2_f1_p = _mm256_permute4x64_pd(delta_w1_2_p_f1, _MM_SHUFFLE(3,1,2,0));

				delta_ws1_p_f1 = _mm256_hadd_pd( perm_1_f1_p, perm_2_f1_p );



				delta_w1_1_p_f2 = _mm256_hadd_pd(delta_w1_p_c0_f2, delta_w1_p_c2_f2);
				delta_w1_2_p_f2 = _mm256_hadd_pd(delta_w1_p_c1_f2, delta_w1_p_c3_f2);

				perm_1_f2_p = _mm256_permute4x64_pd(delta_w1_1_p_f2, _MM_SHUFFLE(3,1,2,0));
				perm_2_f2_p = _mm256_permute4x64_pd(delta_w1_2_p_f2, _MM_SHUFFLE(3,1,2,0));

				delta_ws1_p_f2 = _mm256_hadd_pd( perm_1_f2_p, perm_2_f2_p );


			    // -----------------------------------------------filter weight update--------------------------------------------

			    // modifying delta_w to account for sign function applied on weights

			    // -----------------------------------------------filter 0-------------------------------------------------------
			    weights1_p_f0   = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f, r, c) );

			    vmask_ones1_f0  = _mm256_cmp_pd(weights1_p_f0, ones_p , 0x02); //LE
			    vmask_mones1_f0 = _mm256_cmp_pd(weights1_p_f0, mones_p, 0x0d); //GE
			    vmask_final1_f0 = _mm256_and_pd(vmask_ones1_f0, vmask_mones1_f0); // if >= -1.00 && <= 1.00


			    if_result1_p_f0 = _mm256_add_pd( _mm256_mul_pd(alphas_p_f0, weights1_p_f0), recip_n_p);

			    delta_w_if1_p_f0   = _mm256_and_pd(vmask_final1_f0, if_result1_p_f0);
				delta_w_else1_p_f0 = _mm256_andnot_pd(vmask_final1_f0, recip_n_p);

				delta_ws1_p_f0     = _mm256_mul_pd( delta_ws1_p_f0, _mm256_or_pd(delta_w_if1_p_f0, delta_w_else1_p_f0) );

				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f  , r, c), _mm256_sub_pd( weights1_p_f0, _mm256_mul_pd(multiplier_p, delta_ws1_p_f0) ) );


				// -----------------------------------------------filter 1-------------------------------------------------------
			    weights1_p_f1 = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f+1, r, c) );

			    vmask_ones1_f1  = _mm256_cmp_pd(weights1_p_f1, ones_p , 0x02); //LE
			    vmask_mones1_f1 = _mm256_cmp_pd(weights1_p_f1, mones_p, 0x0d); //GE
			    vmask_final1_f1 = _mm256_and_pd(vmask_ones1_f1, vmask_mones1_f1); // if >= -1.00 && <= 1.00


			    if_result1_p_f1 = _mm256_add_pd( _mm256_mul_pd(alphas_p_f1, weights1_p_f1), recip_n_p);

			    delta_w_if1_p_f1   = _mm256_and_pd(vmask_final1_f1, if_result1_p_f1);
				delta_w_else1_p_f1 = _mm256_andnot_pd(vmask_final1_f1, recip_n_p);

				delta_ws1_p_f1     = _mm256_mul_pd( delta_ws1_p_f1, _mm256_or_pd(delta_w_if1_p_f1, delta_w_else1_p_f1) );

				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f+1, r, c), _mm256_sub_pd( weights1_p_f1, _mm256_mul_pd(multiplier_p, delta_ws1_p_f1) ) );


				// -----------------------------------------------filter 2-------------------------------------------------------
			    weights1_p_f2 = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f+2, r, c) );

			    vmask_ones1_f2  = _mm256_cmp_pd(weights1_p_f2, ones_p , 0x02); //LE
			    vmask_mones1_f2 = _mm256_cmp_pd(weights1_p_f2, mones_p, 0x0d); //GE
			    vmask_final1_f2 = _mm256_and_pd(vmask_ones1_f2, vmask_mones1_f2); // if >= -1.00 && <= 1.00


			    if_result1_p_f2 = _mm256_add_pd( _mm256_mul_pd(alphas_p_f2, weights1_p_f2), recip_n_p);

			    delta_w_if1_p_f2   = _mm256_and_pd(vmask_final1_f2, if_result1_p_f2);
				delta_w_else1_p_f2 = _mm256_andnot_pd(vmask_final1_f2, recip_n_p);

				delta_ws1_p_f2     = _mm256_mul_pd( delta_ws1_p_f2, _mm256_or_pd(delta_w_if1_p_f2, delta_w_else1_p_f2) );

				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f+2, r, c), _mm256_sub_pd( weights1_p_f2, _mm256_mul_pd(multiplier_p, delta_ws1_p_f2) ) );
			}

			// elements remaining at the end of the current row of the filter
			for (; c < FIL_COLS; ++c)
			{

			    delta_w2_p_f0 = _mm256_set1_pd(0.0);
			    delta_w2_p_f1 = _mm256_set1_pd(0.0);
			    delta_w2_p_f2 = _mm256_set1_pd(0.0);

				delta_w2_rem_f0 = 0.0;
				delta_w2_rem_f1 = 0.0;
				delta_w2_rem_f2 = 0.0;

			    for (int b = 0; b < BATCH_SIZE; ++b)
			    {

			    	cur_image2 = shuffle_index[b+base];

	    			for (int i = 0; i < N_ROWS_CONV; ++i)
	    			{

	    				for (j = 0; j+3 < N_COLS_CONV; j=j+4)
	    				{
	    					INCREMENT_FLOPS(36)

	    					    conv_val2_p_f0 = _mm256_loadu_pd( (conv_t->data)       + ind_conv_out(b, f  , i, j) );
	    					    conv_val2_p_f1 = _mm256_loadu_pd( (conv_t->data)       + ind_conv_out(b, f+1, i, j) );
	    					    conv_val2_p_f2 = _mm256_loadu_pd( (conv_t->data)       + ind_conv_out(b, f+2, i, j) );

	    					del_conv_val2_p_f0 = _mm256_loadu_pd( (del_conv->data)     + ind_conv_out(b, f  , i, j) );
	    					del_conv_val2_p_f1 = _mm256_loadu_pd( (del_conv->data)     + ind_conv_out(b, f+1, i, j) );
	    					del_conv_val2_p_f2 = _mm256_loadu_pd( (del_conv->data)     + ind_conv_out(b, f+2, i, j) );

	    					input_pixel2_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image2, i+r, j+c) );


	    					// ----------------------------------------derivative of ReLU--------------------------------------------------

	    					// ---------------------------------------------filter 0-------------------------------------------------------
	    					    vmask2_f0 = _mm256_cmp_pd(    conv_val2_p_f0, zeroes_p, 0x0e); // 0x0e => GT.
	    					    and2_p_f0 = _mm256_and_pd(del_conv_val2_p_f0,       vmask2_f0); // zero out the del_convs for which convs are less than equal to zero
	    					    mul2_p_f0 = _mm256_mul_pd(         and2_p_f0, input_pixel2_p);
	    					delta_w2_p_f0 = _mm256_add_pd(     delta_w2_p_f0,         mul2_p_f0);

	    					// ---------------------------------------------filter 1-------------------------------------------------------
	    					    vmask2_f1 = _mm256_cmp_pd(    conv_val2_p_f1, zeroes_p, 0x0e); // 0x0e => GT.
	    					    and2_p_f1 = _mm256_and_pd(del_conv_val2_p_f1,       vmask2_f1); // zero out the del_convs for which convs are less than equal to zero
	    					    mul2_p_f1 = _mm256_mul_pd(         and2_p_f1, input_pixel2_p);
	    					delta_w2_p_f1 = _mm256_add_pd(     delta_w2_p_f1,         mul2_p_f1);

	    					// ---------------------------------------------filter 2-------------------------------------------------------
	    					    vmask2_f2 = _mm256_cmp_pd(    conv_val2_p_f2, zeroes_p, 0x0e); // 0x0e => GT.
	    					    and2_p_f2 = _mm256_and_pd(del_conv_val2_p_f2,       vmask2_f2); // zero out the del_convs for which convs are less than equal to zero
	    					    mul2_p_f2 = _mm256_mul_pd(         and2_p_f2, input_pixel2_p);
	    					delta_w2_p_f2 = _mm256_add_pd(     delta_w2_p_f2,         mul2_p_f2);
	    				}
	    			}
			    }

			    INCREMENT_FLOPS(33)

			    delta_w2_f0 = ( delta_w2_p_f0[0] + delta_w2_p_f0[1] + delta_w2_p_f0[2] + delta_w2_p_f0[3] );
			    delta_w2_f1 = ( delta_w2_p_f1[0] + delta_w2_p_f1[1] + delta_w2_p_f1[2] + delta_w2_p_f1[3] );
			    delta_w2_f2 = ( delta_w2_p_f2[0] + delta_w2_p_f2[1] + delta_w2_p_f2[2] + delta_w2_p_f2[3] );


			    // modifying delta_w to account for sign function applied on weights
			    weight2_f0 = (fil_w->data)[ind_fil_w(f  , r, c)];
			    weight2_f1 = (fil_w->data)[ind_fil_w(f+1, r, c)];
			    weight2_f2 = (fil_w->data)[ind_fil_w(f+2, r, c)];

			    // -----------------------------------------------filter 0--------------------------------------------
			    if (weight2_f0 <= 1.0 && weight2_f0 >= -1.0)
			    {
			    	delta_w2_f0 *= ( (alphas[f  ] * weight2_f0) +  recip_n );
			    }
			    else
		    	{
		    		delta_w2_f0 *= recip_n;
		    	}

		    	// -----------------------------------------------filter 1--------------------------------------------
			    if (weight2_f1 <= 1.0 && weight2_f1 >= -1.0)
			    {
			    	delta_w2_f1 *= ( (alphas[f+1] * weight2_f1) +  recip_n );
			    }
			    else
		    	{
		    		delta_w2_f1 *= recip_n;
		    	}

		    	// -----------------------------------------------filter 2--------------------------------------------
			    if (weight2_f2 <= 1.0 && weight2_f2 >= -1.0)
			    {
			    	delta_w2_f2 *= ( (alphas[f+2] * weight2_f2) +  recip_n );
			    }
			    else
		    	{
		    		delta_w2_f2 *= recip_n;
		    	}


			    (fil_w->data)[ind_fil_w(f  , r, c)] = weight2_f0 - MULTIPLIER * delta_w2_f0;
			    (fil_w->data)[ind_fil_w(f+1, r, c)] = weight2_f1 - MULTIPLIER * delta_w2_f1;
			    (fil_w->data)[ind_fil_w(f+2, r, c)] = weight2_f2 - MULTIPLIER * delta_w2_f2;
			}
		}
	}

	for (; f < NUM_FILS; ++f)
	{

		alphas_p = _mm256_set1_pd(alphas[f]);

		for (int r = 0; r < FIL_ROWS; ++r)
		{

			for (c = 0; c+3 < FIL_COLS; c=c+4)
			{

			    delta_w3_p_c0  = _mm256_set1_pd(0.0);
			    delta_w3_p_c1  = _mm256_set1_pd(0.0);
			    delta_w3_p_c2  = _mm256_set1_pd(0.0);
			    delta_w3_p_c3  = _mm256_set1_pd(0.0);
			    delta_w3_p_rem = _mm256_set1_pd(0.0);

			    for (int b = 0; b < BATCH_SIZE; ++b)
			    {

			    	cur_image3 = shuffle_index[b+base];

	    			for (int i = 0; i < N_ROWS_CONV; ++i)
	    			{

	    				prev_c1_p = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image3, i+r, c  ) );
	    				prev_c2_p = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image3, i+r, c+1) );
	    				prev_c3_p = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image3, i+r, c+2) );

	    				for (j = 0; j+3 < N_COLS_CONV; j=j+4)
	    				{
	    					INCREMENT_FLOPS(36)

	    					conv_val3_p     = _mm256_loadu_pd( (conv_t->data)       + ind_conv_out(b, f, i, j) );
	    					del_conv_val3_p = _mm256_loadu_pd( (del_conv->data)     + ind_conv_out(b, f, i, j) );

	    					input_pixel3_p_c0  = prev_c1_p;
	    					input_pixel3_p_c1  = prev_c2_p;
	    					input_pixel3_p_c2  = prev_c3_p;
	    					input_pixel3_p_c3  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image3, i+r, j+c+3) );


	    					// applying ReLU
	    					vmask3 = _mm256_cmp_pd(conv_val3_p, zeroes_p, 0x0e); // 0x0e => GT.
	    					and3_p = _mm256_and_pd(del_conv_val3_p, vmask3); // zero out the del_convs for which convs are less than equal to zero

	    					mul3_p_c0     = _mm256_mul_pd(and3_p, input_pixel3_p_c0);
	    					mul3_p_c1     = _mm256_mul_pd(and3_p, input_pixel3_p_c1);
	    					mul3_p_c2     = _mm256_mul_pd(and3_p, input_pixel3_p_c2);
	    					mul3_p_c3     = _mm256_mul_pd(and3_p, input_pixel3_p_c3);

	    					delta_w3_p_c0 = _mm256_add_pd(delta_w3_p_c0, mul3_p_c0);
	    					delta_w3_p_c1 = _mm256_add_pd(delta_w3_p_c1, mul3_p_c1);
	    					delta_w3_p_c2 = _mm256_add_pd(delta_w3_p_c2, mul3_p_c2);
	    					delta_w3_p_c3 = _mm256_add_pd(delta_w3_p_c3, mul3_p_c3);


	    					// storing previously loaded pixels
	    					prev_c1_p = input_pixel3_p_c1;
	    					prev_c2_p = input_pixel3_p_c2;
	    					prev_c3_p = input_pixel3_p_c3;
	    				}
	    			}
			    }

			    INCREMENT_FLOPS(44)

			    delta_w3_1_p = _mm256_hadd_pd(delta_w3_p_c0, delta_w3_p_c2);
				delta_w3_2_p = _mm256_hadd_pd(delta_w3_p_c1, delta_w3_p_c3);

				perm_1_p = _mm256_permute4x64_pd(delta_w3_1_p, _MM_SHUFFLE(3,1,2,0));
				perm_2_p = _mm256_permute4x64_pd(delta_w3_2_p, _MM_SHUFFLE(3,1,2,0));

				delta_ws3_p = _mm256_hadd_pd( perm_1_p, perm_2_p );

			    // -----------------------------------------------filter weight update--------------------------------------------

			    // modifying delta_w to account for sign function applied on weights
			    weights3_p = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f, r, c) );

			    vmask_ones3  = _mm256_cmp_pd(weights3_p, ones_p , 0x02); //LE
			    vmask_mones3 = _mm256_cmp_pd(weights3_p, mones_p, 0x0d); //GE
			    vmask_final3 = _mm256_and_pd(vmask_ones3, vmask_mones3); // if >= -1.00 && <= 1.00


			    if_result3_p = _mm256_add_pd( _mm256_mul_pd(alphas_p, weights3_p), recip_n_p);

			    delta_w_if3_p   = _mm256_and_pd(vmask_final3, if_result3_p);
				delta_w_else3_p = _mm256_andnot_pd(vmask_final3, recip_n_p);

				delta_ws3_p     = _mm256_mul_pd( delta_ws3_p, _mm256_or_pd(delta_w_if3_p, delta_w_else3_p) );

				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f, r, c), _mm256_sub_pd( weights3_p, _mm256_mul_pd(multiplier_p, delta_ws3_p) ) );
			}

			// elements remaining at the end of the current row of the filter
			for (; c < FIL_COLS; ++c)
			{

			    delta_w4_p = _mm256_set1_pd(0.0);

				delta_w4_rem = 0.0;

			    for (int b = 0; b < BATCH_SIZE; ++b)
			    {

			    	cur_image4 = shuffle_index[b+base];

	    			for (int i = 0; i < N_ROWS_CONV; ++i)
	    			{

	    				for (j = 0; j+3 < N_COLS_CONV; j=j+4)
	    				{
	    					INCREMENT_FLOPS(12)

	    					conv_val4_p     = _mm256_loadu_pd( (conv_t->data)       + ind_conv_out(b, f, i, j) );
	    					del_conv_val4_p = _mm256_loadu_pd( (del_conv->data)     + ind_conv_out(b, f, i, j) );

	    					input_pixel4_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image4, i+r, j+c) );


	    					// applying ReLU
	    					vmask4 = _mm256_cmp_pd(conv_val4_p, zeroes_p, 0x0e); // 0x0e => GT.
	    					and4_p = _mm256_and_pd(del_conv_val4_p, vmask4); // zero out the del_convs for which convs are less than equal to zero
	    					mul4_p = _mm256_mul_pd(and4_p, input_pixel4_p);
	    					delta_w4_p = _mm256_add_pd(delta_w4_p, mul4_p);
	    				}
	    			}
			    }

			    INCREMENT_FLOPS(11)

			    delta_w4 = ( delta_w4_p[0] + delta_w4_p[1] + delta_w4_p[2] + delta_w4_p[3] );


			    // modifying delta_w to account for sign function applied on weights
			    weight4 = (fil_w->data)[ind_fil_w(f, r, c)];

			    // -----------------------------------------------filter 0--------------------------------------------
			    if (weight4 <= 1.0 && weight4 >= -1.0)
			    {
			    	delta_w4 *= ( (alphas[f]*weight4) +  recip_n );
			    }
			    else
		    	{
		    		delta_w4 *= recip_n;
		    	}


			    (fil_w->data)[ind_fil_w(f, r, c)] = weight4 - MULTIPLIER * delta_w4;
			}
		}
	}
}*/

// Vectorized: fill rows and cols, conv cols unrolled; hadds used; best perf
void bin_update_conv_weights(tensor* fil_w, tensor* fil_bin_w, double alphas[NUM_FILS], tensor* del_conv, tensor* conv_t,
								tensor* input_images, int base, int shuffle_index[])
{

	INCREMENT_FLOPS(1)

	double recip_n    = 1.0/(FIL_ROWS * FIL_COLS);
	__m256d recip_n_p = _mm256_set1_pd( recip_n );

	__m256d multiplier_p = _mm256_set1_pd(MULTIPLIER);

	int cur_image;

	double delta_w, alpha, weight;

	__m256d zeroes_p = _mm256_set1_pd(0.0);
	__m256d delta_w_p;
	__m256d conv_val_p;
	__m256d del_conv_val_p;
	__m256d input_pixel_p;

	__m256d vmask;
	__m256d and_p;
	__m256d ones_p  = _mm256_set1_pd(1.0);
	__m256d mones_p = _mm256_set1_pd(-1.0);
	__m256d vmask_ones, vmask_mones, vmask_final;
	__m256d weight_p, alphas_p;

	__m256d if_result_p;
	__m256d delta_w_if_p;
	__m256d delta_w_else_p;


	__m256d delta_w_c0_p;
	__m256d delta_w_c1_p;
	__m256d delta_w_c2_p;
	__m256d delta_w_c3_p;

	__m256d input_pixel_c0_p;
	__m256d input_pixel_c1_p;
	__m256d input_pixel_c2_p;
	__m256d input_pixel_c3_p;

	__m256d delta_w_1_p;
	__m256d delta_w_2_p;

	__m256d perm_1_p;
	__m256d perm_2_p;

	__m256d delta_w_final_p;

	int c, j, r;

	__m256d delta_w_r0_c0_p;
	__m256d delta_w_r0_c1_p;
	__m256d delta_w_r0_c2_p;
	__m256d delta_w_r0_c3_p;

	__m256d delta_w_r1_c0_p;
	__m256d delta_w_r1_c1_p;
	__m256d delta_w_r1_c2_p;
	__m256d delta_w_r1_c3_p;

	__m256d delta_w_r2_c0_p;
	__m256d delta_w_r2_c1_p;
	__m256d delta_w_r2_c2_p;
	__m256d delta_w_r2_c3_p;

	__m256d delta_w_r3_c0_p;
	__m256d delta_w_r3_c1_p;
	__m256d delta_w_r3_c2_p;
	__m256d delta_w_r3_c3_p;


	__m256d input_pixel_r0_c0_p;
	__m256d input_pixel_r0_c1_p;
	__m256d input_pixel_r0_c2_p;
	__m256d input_pixel_r0_c3_p;

	__m256d input_pixel_r1_c0_p;
	__m256d input_pixel_r1_c1_p;
	__m256d input_pixel_r1_c2_p;
	__m256d input_pixel_r1_c3_p;

	__m256d input_pixel_r2_c0_p;
	__m256d input_pixel_r2_c1_p;
	__m256d input_pixel_r2_c2_p;
	__m256d input_pixel_r2_c3_p;

	__m256d input_pixel_r3_c0_p;
	__m256d input_pixel_r3_c1_p;
	__m256d input_pixel_r3_c2_p;
	__m256d input_pixel_r3_c3_p;


	__m256d delta_w_1_r0_p;
	__m256d delta_w_2_r0_p;

	__m256d delta_w_1_r1_p;
	__m256d delta_w_2_r1_p;

	__m256d delta_w_1_r2_p;
	__m256d delta_w_2_r2_p;

	__m256d delta_w_1_r3_p;
	__m256d delta_w_2_r3_p;


	__m256d perm_1_r0_p;
	__m256d perm_2_r0_p;

	__m256d perm_1_r1_p;
	__m256d perm_2_r1_p;

	__m256d perm_1_r2_p;
	__m256d perm_2_r2_p;

	__m256d perm_1_r3_p;
	__m256d perm_2_r3_p;

	__m256d delta_w_final_r0_p;
	__m256d delta_w_final_r1_p;
	__m256d delta_w_final_r2_p;
	__m256d delta_w_final_r3_p;


	__m256d weight_r0_p;
	__m256d weight_r1_p;
	__m256d weight_r2_p;
	__m256d weight_r3_p;

	__m256d vmask_ones_r0;
	__m256d vmask_ones_r1;
	__m256d vmask_ones_r2;
	__m256d vmask_ones_r3;

	__m256d vmask_mones_r0;
	__m256d vmask_mones_r1;
	__m256d vmask_mones_r2;
	__m256d vmask_mones_r3;

	__m256d vmask_final_r0;
	__m256d vmask_final_r1;
	__m256d vmask_final_r2;
	__m256d vmask_final_r3;

	__m256d if_result_r0_p;
	__m256d if_result_r1_p;
	__m256d if_result_r2_p;
	__m256d if_result_r3_p;

	__m256d delta_w_if_r0_p;
	__m256d delta_w_if_r1_p;
	__m256d delta_w_if_r2_p;
	__m256d delta_w_if_r3_p;

	__m256d delta_w_else_r0_p;
	__m256d delta_w_else_r1_p;
	__m256d delta_w_else_r2_p;
	__m256d delta_w_else_r3_p;


	for (int f = 0; f < NUM_FILS; ++f)
	{

		alpha = alphas[f];
		alphas_p = _mm256_set1_pd(alpha);

		for (r = 0; r+3 < FIL_ROWS; r=r+4)
		{
			for (c = 0; c+3 < FIL_COLS; c=c+4)
			{

			    delta_w_r0_c0_p = _mm256_set1_pd(0.0);
			    delta_w_r0_c1_p = _mm256_set1_pd(0.0);
			    delta_w_r0_c2_p = _mm256_set1_pd(0.0);
			    delta_w_r0_c3_p = _mm256_set1_pd(0.0);

				delta_w_r1_c0_p = _mm256_set1_pd(0.0);
				delta_w_r1_c1_p = _mm256_set1_pd(0.0);
				delta_w_r1_c2_p = _mm256_set1_pd(0.0);
				delta_w_r1_c3_p = _mm256_set1_pd(0.0);

				delta_w_r2_c0_p = _mm256_set1_pd(0.0);
				delta_w_r2_c1_p = _mm256_set1_pd(0.0);
				delta_w_r2_c2_p = _mm256_set1_pd(0.0);
				delta_w_r2_c3_p = _mm256_set1_pd(0.0);

				delta_w_r3_c0_p = _mm256_set1_pd(0.0);
				delta_w_r3_c1_p = _mm256_set1_pd(0.0);
				delta_w_r3_c2_p = _mm256_set1_pd(0.0);
				delta_w_r3_c3_p = _mm256_set1_pd(0.0);

			    for (int b = 0; b < BATCH_SIZE; ++b)
			    {

			    	cur_image = shuffle_index[b+base];
	    			for (int i = 0; i < N_ROWS_CONV; ++i)
	    			{

	    				for (int j = 0; j+3 < N_COLS_CONV; j=j+4)
	    				{

	    					INCREMENT_FLOPS(3)

	    					conv_val_p     = _mm256_loadu_pd(   (conv_t->data) + ind_conv_out(b, f, i, j) );
	    					del_conv_val_p = _mm256_loadu_pd( (del_conv->data) + ind_conv_out(b, f, i, j) );

	    					input_pixel_r0_c0_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r  , j+c  ) );
	    					input_pixel_r0_c1_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r  , j+c+1) );
	    					input_pixel_r0_c2_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r  , j+c+2) );
	    					input_pixel_r0_c3_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r  , j+c+3) );

	    					input_pixel_r1_c0_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+1, j+c  ) );
	    					input_pixel_r1_c1_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+1, j+c+1) );
	    					input_pixel_r1_c2_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+1, j+c+2) );
	    					input_pixel_r1_c3_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+1, j+c+3) );

	    					input_pixel_r2_c0_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+2, j+c  ) );
	    					input_pixel_r2_c1_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+2, j+c+1) );
	    					input_pixel_r2_c2_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+2, j+c+2) );
	    					input_pixel_r2_c3_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+2, j+c+3) );

	    					input_pixel_r3_c0_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+3, j+c  ) );
	    					input_pixel_r3_c1_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+3, j+c+1) );
	    					input_pixel_r3_c2_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+3, j+c+2) );
	    					input_pixel_r3_c3_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r+3, j+c+3) );

	    					// derivative of ReLU
	    					vmask = _mm256_cmp_pd( conv_val_p, zeroes_p, 0x0e ); // 0x0e => GT
	    					and_p = _mm256_and_pd( del_conv_val_p, and_p );

	    					delta_w_r0_c0_p = _mm256_add_pd( delta_w_r0_c0_p, _mm256_mul_pd( and_p, input_pixel_r0_c0_p ) );
	    					delta_w_r0_c1_p = _mm256_add_pd( delta_w_r0_c1_p, _mm256_mul_pd( and_p, input_pixel_r0_c1_p ) );
	    					delta_w_r0_c2_p = _mm256_add_pd( delta_w_r0_c2_p, _mm256_mul_pd( and_p, input_pixel_r0_c2_p ) );
	    					delta_w_r0_c3_p = _mm256_add_pd( delta_w_r0_c3_p, _mm256_mul_pd( and_p, input_pixel_r0_c3_p ) );

	    					delta_w_r1_c0_p = _mm256_add_pd( delta_w_r1_c0_p, _mm256_mul_pd( and_p, input_pixel_r1_c0_p ) );
	    					delta_w_r1_c1_p = _mm256_add_pd( delta_w_r1_c1_p, _mm256_mul_pd( and_p, input_pixel_r1_c1_p ) );
	    					delta_w_r1_c2_p = _mm256_add_pd( delta_w_r1_c2_p, _mm256_mul_pd( and_p, input_pixel_r1_c2_p ) );
	    					delta_w_r1_c3_p = _mm256_add_pd( delta_w_r1_c3_p, _mm256_mul_pd( and_p, input_pixel_r1_c3_p ) );

	    					delta_w_r2_c0_p = _mm256_add_pd( delta_w_r2_c0_p, _mm256_mul_pd( and_p, input_pixel_r2_c0_p ) );
	    					delta_w_r2_c1_p = _mm256_add_pd( delta_w_r2_c1_p, _mm256_mul_pd( and_p, input_pixel_r2_c1_p ) );
	    					delta_w_r2_c2_p = _mm256_add_pd( delta_w_r2_c2_p, _mm256_mul_pd( and_p, input_pixel_r2_c2_p ) );
	    					delta_w_r2_c3_p = _mm256_add_pd( delta_w_r2_c3_p, _mm256_mul_pd( and_p, input_pixel_r2_c3_p ) );

	    					delta_w_r3_c0_p = _mm256_add_pd( delta_w_r3_c0_p, _mm256_mul_pd( and_p, input_pixel_r3_c0_p ) );
	    					delta_w_r3_c1_p = _mm256_add_pd( delta_w_r3_c1_p, _mm256_mul_pd( and_p, input_pixel_r3_c1_p ) );
	    					delta_w_r3_c2_p = _mm256_add_pd( delta_w_r3_c2_p, _mm256_mul_pd( and_p, input_pixel_r3_c2_p ) );
	    					delta_w_r3_c3_p = _mm256_add_pd( delta_w_r3_c3_p, _mm256_mul_pd( and_p, input_pixel_r3_c3_p ) );
	    				}
	    			}
			    }

			    INCREMENT_FLOPS(7)


			    // horizontally sum the four vectors
				delta_w_1_r0_p = _mm256_hadd_pd(delta_w_r0_c0_p, delta_w_r0_c2_p);
				delta_w_2_r0_p = _mm256_hadd_pd(delta_w_r0_c1_p, delta_w_r0_c3_p);

				perm_1_r0_p = _mm256_permute4x64_pd(delta_w_1_r0_p, _MM_SHUFFLE(3,1,2,0));
				perm_2_r0_p = _mm256_permute4x64_pd(delta_w_2_r0_p, _MM_SHUFFLE(3,1,2,0));

				delta_w_final_r0_p = _mm256_hadd_pd( perm_1_r0_p, perm_2_r0_p );


				delta_w_1_r1_p = _mm256_hadd_pd(delta_w_r1_c0_p, delta_w_r1_c2_p);
				delta_w_2_r1_p = _mm256_hadd_pd(delta_w_r1_c1_p, delta_w_r1_c3_p);

				perm_1_r1_p = _mm256_permute4x64_pd(delta_w_1_r1_p, _MM_SHUFFLE(3,1,2,0));
				perm_2_r1_p = _mm256_permute4x64_pd(delta_w_2_r1_p, _MM_SHUFFLE(3,1,2,0));

				delta_w_final_r1_p = _mm256_hadd_pd( perm_1_r1_p, perm_2_r1_p );


				delta_w_1_r2_p = _mm256_hadd_pd(delta_w_r2_c0_p, delta_w_r2_c2_p);
				delta_w_2_r2_p = _mm256_hadd_pd(delta_w_r2_c1_p, delta_w_r2_c3_p);

				perm_1_r2_p = _mm256_permute4x64_pd(delta_w_1_r2_p, _MM_SHUFFLE(3,1,2,0));
				perm_2_r2_p = _mm256_permute4x64_pd(delta_w_2_r2_p, _MM_SHUFFLE(3,1,2,0));

				delta_w_final_r2_p = _mm256_hadd_pd( perm_1_r2_p, perm_2_r2_p );


				delta_w_1_r3_p = _mm256_hadd_pd(delta_w_r3_c0_p, delta_w_r3_c2_p);
				delta_w_2_r3_p = _mm256_hadd_pd(delta_w_r3_c1_p, delta_w_r3_c3_p);

				perm_1_r3_p = _mm256_permute4x64_pd(delta_w_1_r3_p, _MM_SHUFFLE(3,1,2,0));
				perm_2_r3_p = _mm256_permute4x64_pd(delta_w_2_r3_p, _MM_SHUFFLE(3,1,2,0));

				delta_w_final_r3_p = _mm256_hadd_pd( perm_1_r3_p, perm_2_r3_p );



			    weight_r0_p = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f, r  , c) );
			    weight_r1_p = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f, r+1, c) );
			    weight_r2_p = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f, r+2, c) );
			    weight_r3_p = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f, r+3, c) );


			       vmask_ones_r0   = _mm256_cmp_pd(weight_r0_p, ones_p , 0x02); //LE
			      vmask_mones_r0   = _mm256_cmp_pd(weight_r0_p, mones_p, 0x0d); //GE
			      vmask_final_r0   = _mm256_and_pd(vmask_ones_r0  , vmask_mones_r0); // if >= -1.00 && <= 1.00
			        if_result_r0_p = _mm256_add_pd( _mm256_mul_pd(alphas_p, weight_r0_p), recip_n_p);
			       delta_w_if_r0_p = _mm256_and_pd(vmask_final_r0, if_result_r0_p);
				 delta_w_else_r0_p = _mm256_andnot_pd(vmask_final_r0, recip_n_p);
				delta_w_final_r0_p = _mm256_mul_pd( delta_w_final_r0_p, _mm256_or_pd(delta_w_if_r0_p, delta_w_else_r0_p) );
				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f, r  , c), _mm256_sub_pd( weight_r0_p, _mm256_mul_pd(multiplier_p, delta_w_final_r0_p) ) );

				   vmask_ones_r1   = _mm256_cmp_pd(weight_r1_p, ones_p , 0x02); //LE
			      vmask_mones_r1   = _mm256_cmp_pd(weight_r1_p, mones_p, 0x0d); //GE
			      vmask_final_r1   = _mm256_and_pd(vmask_ones_r1  , vmask_mones_r1); // if >= -1.00 && <= 1.00
			        if_result_r1_p = _mm256_add_pd( _mm256_mul_pd(alphas_p, weight_r1_p), recip_n_p);
			       delta_w_if_r1_p = _mm256_and_pd(vmask_final_r1, if_result_r1_p);
				 delta_w_else_r1_p = _mm256_andnot_pd(vmask_final_r1, recip_n_p);
				delta_w_final_r1_p = _mm256_mul_pd( delta_w_final_r1_p, _mm256_or_pd(delta_w_if_r1_p, delta_w_else_r1_p) );
				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f, r+1, c), _mm256_sub_pd( weight_r1_p, _mm256_mul_pd(multiplier_p, delta_w_final_r1_p) ) );

				   vmask_ones_r2   = _mm256_cmp_pd(weight_r2_p, ones_p , 0x02); //LE
			      vmask_mones_r2   = _mm256_cmp_pd(weight_r2_p, mones_p, 0x0d); //GE
			      vmask_final_r2   = _mm256_and_pd(vmask_ones_r2  , vmask_mones_r2); // if >= -1.00 && <= 1.00
			        if_result_r2_p = _mm256_add_pd( _mm256_mul_pd(alphas_p, weight_r2_p), recip_n_p);
			       delta_w_if_r2_p = _mm256_and_pd(vmask_final_r2, if_result_r2_p);
				 delta_w_else_r2_p = _mm256_andnot_pd(vmask_final_r2, recip_n_p);
				delta_w_final_r2_p = _mm256_mul_pd( delta_w_final_r2_p, _mm256_or_pd(delta_w_if_r2_p, delta_w_else_r2_p) );
				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f, r+2, c), _mm256_sub_pd( weight_r2_p, _mm256_mul_pd(multiplier_p, delta_w_final_r2_p) ) );

				   vmask_ones_r3   = _mm256_cmp_pd(weight_r3_p, ones_p , 0x02); //LE
			      vmask_mones_r3   = _mm256_cmp_pd(weight_r3_p, mones_p, 0x0d); //GE
			      vmask_final_r3   = _mm256_and_pd(vmask_ones_r3  , vmask_mones_r3); // if >= -1.00 && <= 1.00
			        if_result_r3_p = _mm256_add_pd( _mm256_mul_pd(alphas_p, weight_r3_p), recip_n_p);
			       delta_w_if_r3_p = _mm256_and_pd(vmask_final_r3, if_result_r3_p);
				 delta_w_else_r3_p = _mm256_andnot_pd(vmask_final_r3, recip_n_p);
				delta_w_final_r3_p = _mm256_mul_pd( delta_w_final_r3_p, _mm256_or_pd(delta_w_if_r3_p, delta_w_else_r3_p) );
				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f, r+3, c), _mm256_sub_pd( weight_r3_p, _mm256_mul_pd(multiplier_p, delta_w_final_r3_p) ) );
			}


			for (; c < FIL_COLS; ++c)
			{

			    delta_w_p = _mm256_set1_pd(0.0);

			    for (int b = 0; b < BATCH_SIZE; ++b)
			    {

			    	cur_image = shuffle_index[b+base];
	    			for (int i = 0; i < N_ROWS_CONV; ++i)
	    			{

	    				for (int j = 0; j+3 < N_COLS_CONV; j=j+4)
	    				{

	    					INCREMENT_FLOPS(3)

	    					conv_val_p     = _mm256_loadu_pd(   (conv_t->data) + ind_conv_out(b, f, i, j) );
	    					del_conv_val_p = _mm256_loadu_pd( (del_conv->data) + ind_conv_out(b, f, i, j) );

	    					input_pixel_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r, j+c) );

	    					// derivative of ReLU
	    					vmask = _mm256_cmp_pd( conv_val_p, zeroes_p, 0x0e ); // 0x0e => GT
	    					and_p = _mm256_and_pd( del_conv_val_p, and_p );

	    					delta_w_p = _mm256_add_pd( delta_w_p, _mm256_mul_pd( del_conv_val_p, input_pixel_p ) );
	    				}
	    			}
			    }

			    delta_w = delta_w_p[0] + delta_w_p[1] + delta_w_p[2] + delta_w_p[3];

			    INCREMENT_FLOPS(7)


			    // modifying delta_w to account for sign function applied on weights

			    weight = (fil_w->data)[ind_fil_w(f, r, c)];
			    if (weight <= 1 && weight >= -1)
			    {
			    	delta_w *= ( (alpha * weight) +  recip_n );
			    }
			    else
		    	{
		    		delta_w *= recip_n;
		    	}

			    (fil_w->data)[ind_fil_w(f, r, c)] = weight - MULTIPLIER*delta_w;
			}
		}

		for (; r < FIL_ROWS; ++r)
		{
			for (c = 0; c+3 < FIL_COLS; c=c+4)
			{

			    delta_w_c0_p = _mm256_set1_pd(0.0);
			    delta_w_c1_p = _mm256_set1_pd(0.0);
			    delta_w_c2_p = _mm256_set1_pd(0.0);
			    delta_w_c3_p = _mm256_set1_pd(0.0);

			    for (int b = 0; b < BATCH_SIZE; ++b)
			    {

			    	cur_image = shuffle_index[b+base];
	    			for (int i = 0; i < N_ROWS_CONV; ++i)
	    			{

	    				for (int j = 0; j+3 < N_COLS_CONV; j=j+4)
	    				{

	    					INCREMENT_FLOPS(3)

	    					conv_val_p     = _mm256_loadu_pd(   (conv_t->data) + ind_conv_out(b, f, i, j) );
	    					del_conv_val_p = _mm256_loadu_pd( (del_conv->data) + ind_conv_out(b, f, i, j) );

	    					input_pixel_c0_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r, j+c  ) );
	    					input_pixel_c1_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r, j+c+1) );
	    					input_pixel_c2_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r, j+c+2) );
	    					input_pixel_c3_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r, j+c+3) );

	    					// derivative of ReLU
	    					vmask = _mm256_cmp_pd( conv_val_p, zeroes_p, 0x0e ); // 0x0e => GT
	    					and_p = _mm256_and_pd( del_conv_val_p, and_p );

	    					delta_w_c0_p = _mm256_add_pd( delta_w_c0_p, _mm256_mul_pd( and_p, input_pixel_c0_p ) );
	    					delta_w_c1_p = _mm256_add_pd( delta_w_c1_p, _mm256_mul_pd( and_p, input_pixel_c1_p ) );
	    					delta_w_c2_p = _mm256_add_pd( delta_w_c2_p, _mm256_mul_pd( and_p, input_pixel_c2_p ) );
	    					delta_w_c3_p = _mm256_add_pd( delta_w_c3_p, _mm256_mul_pd( and_p, input_pixel_c3_p ) );
	    				}
	    			}
			    }

			    INCREMENT_FLOPS(7)


			    // horizontally sum the four vectors
				delta_w_1_p = _mm256_hadd_pd(delta_w_c0_p, delta_w_c2_p);
				delta_w_2_p = _mm256_hadd_pd(delta_w_c1_p, delta_w_c3_p);

				perm_1_p = _mm256_permute4x64_pd(delta_w_1_p, _MM_SHUFFLE(3,1,2,0));
				perm_2_p = _mm256_permute4x64_pd(delta_w_2_p, _MM_SHUFFLE(3,1,2,0));

				delta_w_final_p = _mm256_hadd_pd( perm_1_p, perm_2_p );


			    weight_p = _mm256_loadu_pd( (fil_w->data) + ind_fil_w(f, r, c) );

			    vmask_ones  = _mm256_cmp_pd(weight_p, ones_p , 0x02); //LE
			    vmask_mones = _mm256_cmp_pd(weight_p, mones_p, 0x0d); //GE

			    vmask_final = _mm256_and_pd(vmask_ones, vmask_mones); // if >= -1.00 && <= 1.00

			    if_result_p = _mm256_add_pd( _mm256_mul_pd(alphas_p, weight_p), recip_n_p);

			    delta_w_if_p   = _mm256_and_pd(vmask_final, if_result_p);
				delta_w_else_p = _mm256_andnot_pd(vmask_final, recip_n_p);

				delta_w_final_p     = _mm256_mul_pd( delta_w_final_p, _mm256_or_pd(delta_w_if_p, delta_w_else_p) );

				_mm256_storeu_pd( (fil_w->data) + ind_fil_w(f, r, c), _mm256_sub_pd( weight_p, _mm256_mul_pd(multiplier_p, delta_w_final_p) ) );
			}


			for (; c < FIL_COLS; ++c)
			{

			    delta_w_p = _mm256_set1_pd(0.0);

			    for (int b = 0; b < BATCH_SIZE; ++b)
			    {

			    	cur_image = shuffle_index[b+base];
	    			for (int i = 0; i < N_ROWS_CONV; ++i)
	    			{

	    				for (int j = 0; j+3 < N_COLS_CONV; j=j+4)
	    				{

	    					INCREMENT_FLOPS(3)

	    					conv_val_p     = _mm256_loadu_pd(   (conv_t->data) + ind_conv_out(b, f, i, j) );
	    					del_conv_val_p = _mm256_loadu_pd( (del_conv->data) + ind_conv_out(b, f, i, j) );

	    					input_pixel_p  = _mm256_loadu_pd( (input_images->data) + ind_input_img(cur_image, i+r, j+c) );

	    					// derivative of ReLU
	    					vmask = _mm256_cmp_pd( conv_val_p, zeroes_p, 0x0e ); // 0x0e => GT
	    					and_p = _mm256_and_pd( del_conv_val_p, and_p );

	    					delta_w_p = _mm256_add_pd( delta_w_p, _mm256_mul_pd( del_conv_val_p, input_pixel_p ) );
	    				}
	    			}
			    }

			    delta_w = delta_w_p[0] + delta_w_p[1] + delta_w_p[2] + delta_w_p[3];

			    INCREMENT_FLOPS(7)


			    // modifying delta_w to account for sign function applied on weights

			    weight = (fil_w->data)[ind_fil_w(f, r, c)];
			    if (weight <= 1 && weight >= -1)
			    {
			    	delta_w *= ( (alpha * weight) +  recip_n );
			    }
			    else
		    	{
		    		delta_w *= recip_n;
		    	}

			    (fil_w->data)[ind_fil_w(f, r, c)] = weight - MULTIPLIER*delta_w;
			}
		}
	}
}

// Vectorized: conv cols unrolled
void update_conv_biases(tensor* fil_b, tensor* del_conv, tensor* conv_t)
{
	double delta_b;
	double conv_val;
	double del_conv_val;

	__m256d zeroes_p = _mm256_set1_pd(0.0);

	__m256d conv_val_i0_p;
	__m256d conv_val_i1_p;
	__m256d conv_val_i2_p;
	__m256d conv_val_i3_p;

	__m256d del_conv_val_i0_p;
	__m256d del_conv_val_i1_p;
	__m256d del_conv_val_i2_p;
	__m256d del_conv_val_i3_p;

	__m256d vmask_i0;
	__m256d vmask_i1;
	__m256d vmask_i2;
	__m256d vmask_i3;

	__m256d and_i0_p;
	__m256d and_i1_p;
	__m256d and_i2_p;
	__m256d and_i3_p;

	__m256d delta_b_p;

	int j;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		delta_b_p = _mm256_set1_pd(0.0);
		delta_b = 0.0;

		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			for (int i = 0; i+3 < N_ROWS_CONV; i=i+4)
			{
				for (j = 0; j+3 < N_COLS_CONV; j=j+4)
				{
					INCREMENT_FLOPS(32)

					conv_val_i0_p = _mm256_loadu_pd( (conv_t->data) + ind_conv_out(b, f, i  , j) );
					conv_val_i1_p = _mm256_loadu_pd( (conv_t->data) + ind_conv_out(b, f, i+1, j) );
					conv_val_i2_p = _mm256_loadu_pd( (conv_t->data) + ind_conv_out(b, f, i+2, j) );
					conv_val_i3_p = _mm256_loadu_pd( (conv_t->data) + ind_conv_out(b, f, i+3, j) );

					del_conv_val_i0_p = _mm256_loadu_pd( (del_conv->data) + ind_conv_out(b, f, i  , j) );
					del_conv_val_i1_p = _mm256_loadu_pd( (del_conv->data) + ind_conv_out(b, f, i+1, j) );
					del_conv_val_i2_p = _mm256_loadu_pd( (del_conv->data) + ind_conv_out(b, f, i+2, j) );
					del_conv_val_i3_p = _mm256_loadu_pd( (del_conv->data) + ind_conv_out(b, f, i+3, j) );

					vmask_i0 = _mm256_cmp_pd(conv_val_i0_p, zeroes_p, 0x0e); // 0x0e => GT
					vmask_i1 = _mm256_cmp_pd(conv_val_i1_p, zeroes_p, 0x0e);
					vmask_i2 = _mm256_cmp_pd(conv_val_i2_p, zeroes_p, 0x0e);
					vmask_i3 = _mm256_cmp_pd(conv_val_i3_p, zeroes_p, 0x0e);

					and_i0_p = _mm256_and_pd(del_conv_val_i0_p, vmask_i0); // 0x0e => GT
					and_i1_p = _mm256_and_pd(del_conv_val_i1_p, vmask_i1);
					and_i2_p = _mm256_and_pd(del_conv_val_i2_p, vmask_i2);
					and_i3_p = _mm256_and_pd(del_conv_val_i3_p, vmask_i3);


					delta_b_p = _mm256_add_pd(delta_b_p, del_conv_val_i0_p);
					delta_b_p = _mm256_add_pd(delta_b_p, del_conv_val_i1_p);
					delta_b_p = _mm256_add_pd(delta_b_p, del_conv_val_i2_p);
					delta_b_p = _mm256_add_pd(delta_b_p, del_conv_val_i3_p);
				}

				for (; j < N_COLS_CONV; ++j)
				{
					INCREMENT_FLOPS(2)

					conv_val = conv_t->data[ind_conv_out(b, f, i, j)];
					del_conv_val = del_conv->data[ind_conv_out(b, f, i, j)];

					if(conv_val > 0.0)
					{
						delta_b	+= del_conv_val;
					}
				}
			}
		}

		INCREMENT_FLOPS(6)
		(fil_b->data)[f] -= MULTIPLIER * (delta_b_p[1] + delta_b_p[1] + delta_b_p[2] + delta_b_p[3] + delta_b);
	}
}
