#include "back_prop.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int N_ROWS_POOL;
int N_COLS_POOL;
int TOTAL_FLOPS;
double MULTIPLIER;

// batches in outermost loop
void update_sotmax_weights(tensor* fully_con_w, tensor* softmax_out, tensor* pool_t, int* labels, int base, int shuffle_index[])
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
}

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

// inner loops on digits and filters unrolled completely
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



	int row_f0;
	int col_f0;
	int row_f1;
	int col_f1;
	int row_f2;
	int col_f2;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		int cur_label = labels[shuffle_index[base+b]];

		softmax_out_0 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 0)];
		softmax_out_1 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 1)];
		softmax_out_2 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 2)];
		softmax_out_3 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 3)];
		softmax_out_4 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 4)];
		softmax_out_5 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 5)];
		softmax_out_6 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 6)];
		softmax_out_7 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 7)];
		softmax_out_8 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 8)];
		softmax_out_9 = (softmax_out->data)[offset(softmax_out, b, 0, 0, 9)];


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

// unroll fil rows and cols, batch in inner loop
void bin_update_conv_weights(tensor* fil_w, tensor* fil_bin_w, double alphas[NUM_FILS], tensor* del_conv, tensor* conv_t, 
								tensor* input_images, int base, int shuffle_index[])
{
	INCREMENT_FLOPS(1)
	double recip_n = 1.0/(FIL_ROWS * FIL_COLS);
	double input_pixel0, input_pixel1, input_pixel2, input_pixel3;
	double conv_val0, conv_val1, conv_val2;
	double del_conv0, del_conv1, del_conv2;
	int cur_image;

	double weight_r0c0f0, weight_r0c0f1, weight_r0c0f2;
	double weight_r0c1f0, weight_r0c1f1, weight_r0c1f2;
	double weight_r1c0f0, weight_r1c0f1, weight_r1c0f2;
	double weight_r1c1f0, weight_r1c1f1, weight_r1c1f2;

	double delta_w_r0c0f0, delta_w_r0c0f1, delta_w_r0c0f2;
    double delta_w_r0c1f0, delta_w_r0c1f1, delta_w_r0c1f2;
    double delta_w_r1c0f0, delta_w_r1c0f1, delta_w_r1c0f2;
    double delta_w_r1c1f0, delta_w_r1c1f1, delta_w_r1c1f2;

    double prev_input_pixel_01, prev_input_pixel_11;


    // declaration for the leftover part, at the end
    double delta_w0, delta_w1, delta_w2;
    double input_pixel;
    double weight0, weight1, weight2;


    int r=0, c=0;
	//unroll outer loop on number of filters
	for (r = 0; r+1 < FIL_ROWS; r=r+2)
	{
		for (c = 0; c+1 < FIL_COLS; c=c+2)
		{
		    delta_w_r0c0f0 = 0.0, delta_w_r0c0f1 = 0.0, delta_w_r0c0f2 = 0.0;
		    delta_w_r0c1f0 = 0.0, delta_w_r0c1f1 = 0.0, delta_w_r0c1f2 = 0.0;
		    delta_w_r1c0f0 = 0.0, delta_w_r1c0f1 = 0.0, delta_w_r1c0f2 = 0.0;
		    delta_w_r1c1f0 = 0.0, delta_w_r1c1f1 = 0.0, delta_w_r1c1f2 = 0.0;

		    for (int b = 0; b < BATCH_SIZE; ++b)
		    {
		    	cur_image = shuffle_index[b+base];

    			for (int i = 0; i < N_ROWS_CONV; ++i)
    			{

    				prev_input_pixel_01 = (input_images->data)[ind_input_img(cur_image, i+r  , 0+c  )];
    				prev_input_pixel_11 = (input_images->data)[ind_input_img(cur_image, i+r+1, 0+c  )];

    				for (int j = 0; j < N_COLS_CONV; ++j)
    				{

    					INCREMENT_FLOPS(27)

    					// one for each r and c, for the unrolling
    					input_pixel0 = prev_input_pixel_01;
    					input_pixel1 = (input_images->data)[ind_input_img(cur_image, i+r  , j+c+1)];
    					input_pixel2 = prev_input_pixel_11;
    					input_pixel3 = (input_images->data)[ind_input_img(cur_image, i+r+1, j+c+1)];

    					// one for each filter
    					conv_val0 = (conv_t->data)[ind_conv_out(b, 0, i, j)];
    					conv_val1 = (conv_t->data)[ind_conv_out(b, 1, i, j)];
    					conv_val2 = (conv_t->data)[ind_conv_out(b, 2, i, j)];

    					// one for each filter
    					del_conv0 = (del_conv->data)[ind_conv_out(b, 0, i, j)];
    					del_conv1 = (del_conv->data)[ind_conv_out(b, 1, i, j)];
    					del_conv2 = (del_conv->data)[ind_conv_out(b, 2, i, j)];
    					
    					// filter 1
    					if( conv_val0 > 0.0 )
    					{
    						delta_w_r0c0f0 += del_conv0*input_pixel0;
    						delta_w_r0c1f0 += del_conv0*input_pixel1;
    						delta_w_r1c0f0 += del_conv0*input_pixel2;
    						delta_w_r1c1f0 += del_conv0*input_pixel3;
    					}

    					// filter 2
    					if( conv_val1 > 0.0 )
    					{
    						delta_w_r0c0f1 += del_conv1*input_pixel0;
    						delta_w_r0c1f1 += del_conv1*input_pixel1;
    						delta_w_r1c0f1 += del_conv1*input_pixel2;
    						delta_w_r1c1f1 += del_conv1*input_pixel3;
    					}

    					// filter 3
    					if( conv_val2 > 0.0 )
    					{
    						delta_w_r0c0f2 += del_conv2*input_pixel0;
    						delta_w_r0c1f2 += del_conv2*input_pixel1;
    						delta_w_r1c0f2 += del_conv2*input_pixel2;
    						delta_w_r1c1f2 += del_conv2*input_pixel3;
    					}

    					prev_input_pixel_01 = input_pixel1;
    					prev_input_pixel_11 = input_pixel3;
    				}
    			}
		    }

		    INCREMENT_FLOPS(60)

		    // modifying delta_w to account for sign function applied on weights
		    weight_r0c0f0 = (fil_w->data)[ind_fil_w(0, r  , c  )];
		    weight_r0c0f1 = (fil_w->data)[ind_fil_w(1, r  , c  )];
		    weight_r0c0f2 = (fil_w->data)[ind_fil_w(2, r  , c  )];

		    weight_r0c1f0 = (fil_w->data)[ind_fil_w(0, r  , c+1)];
		    weight_r0c1f1 = (fil_w->data)[ind_fil_w(1, r  , c+1)];
		    weight_r0c1f2 = (fil_w->data)[ind_fil_w(2, r  , c+1)];

		    weight_r1c0f0 = (fil_w->data)[ind_fil_w(0, r+1, c  )];
		    weight_r1c0f1 = (fil_w->data)[ind_fil_w(1, r+1, c  )];
		    weight_r1c0f2 = (fil_w->data)[ind_fil_w(2, r+1, c  )];

		    weight_r1c1f0 = (fil_w->data)[ind_fil_w(0, r+1, c+1)];
		    weight_r1c1f1 = (fil_w->data)[ind_fil_w(1, r+1, c+1)];
		    weight_r1c1f2 = (fil_w->data)[ind_fil_w(2, r+1, c+1)];

		    // filter 1
		    if (weight_r0c0f0 <= 1.0 && weight_r0c0f0 >= -1.0)
		    {
		    	delta_w_r0c0f0 *= ( (alphas[0]*weight_r0c0f0) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r0c0f0 *= recip_n;
	    	}

	    	if (weight_r0c1f0 <= 1.0 && weight_r0c1f0 >= -1.0)
		    {
		    	delta_w_r0c1f0 *= ( (alphas[0]*weight_r0c1f0) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r0c1f0 *= recip_n;
	    	}

	    	if (weight_r1c0f0 <= 1.0 && weight_r1c0f0 >= -1.0)
		    {
		    	delta_w_r1c0f0 *= ( (alphas[0]*weight_r1c0f0) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r1c0f0 *= recip_n;
	    	}

	    	if (weight_r1c1f0 <= 1.0 && weight_r1c1f0 >= -1.0)
		    {
		    	delta_w_r1c1f0 *= ( (alphas[0]*weight_r1c1f0) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r1c1f0 *= recip_n;
	    	}

	    	// filter 2
	    	if (weight_r0c0f1 <= 1.0 && weight_r0c0f1 >= -1.0)
		    {
		    	delta_w_r0c0f1 *= ( (alphas[1]*weight_r0c0f1) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r0c0f1 *= recip_n;
	    	}

	    	if (weight_r0c1f1 <= 1.0 && weight_r0c1f1 >= -1.0)
		    {
		    	delta_w_r0c1f1 *= ( (alphas[1]*weight_r0c1f1) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r0c1f1 *= recip_n;
	    	}

	    	if (weight_r1c0f1 <= 1.0 && weight_r1c0f1 >= -1.0)
		    {
		    	delta_w_r1c0f1 *= ( (alphas[1]*weight_r1c0f1) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r1c0f1 *= recip_n;
	    	}

	    	if (weight_r1c1f1 <= 1.0 && weight_r1c1f1 >= -1.0)
		    {
		    	delta_w_r1c1f1 *= ( (alphas[1]*weight_r1c1f1) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r1c1f1 *= recip_n;
	    	}

	    	// filter 3
	    	if (weight_r0c0f2 <= 1.0 && weight_r0c0f2 >= -1.0)
		    {
		    	delta_w_r0c0f2 *= ( (alphas[2]*weight_r0c0f2) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r0c0f2 *= recip_n;
	    	}

	    	if (weight_r0c1f2 <= 1.0 && weight_r0c1f2 >= -1.0)
		    {
		    	delta_w_r0c1f2 *= ( (alphas[2]*weight_r0c1f2) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r0c1f2 *= recip_n;
	    	}

	    	if (weight_r1c0f2 <= 1.0 && weight_r1c0f2 >= -1.0)
		    {
		    	delta_w_r1c0f2 *= ( (alphas[2]*weight_r1c0f2) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r1c0f2 *= recip_n;
	    	}

	    	if (weight_r1c1f2 <= 1.0 && weight_r1c1f2 >= -1.0)
		    {
		    	delta_w_r1c1f2 *= ( (alphas[2]*weight_r1c1f2) +  recip_n );
		    }
		    else
	    	{
	    		delta_w_r1c1f2 *= recip_n;
	    	}
		    
	    	INCREMENT_FLOPS(24)

		    (fil_w->data)[ind_fil_w(0, r  , c  )] = weight_r0c0f0 - MULTIPLIER*delta_w_r0c0f0;
		    (fil_w->data)[ind_fil_w(1, r  , c  )] = weight_r0c0f1 - MULTIPLIER*delta_w_r0c0f1;
		    (fil_w->data)[ind_fil_w(2, r  , c  )] = weight_r0c0f2 - MULTIPLIER*delta_w_r0c0f2;

		    (fil_w->data)[ind_fil_w(0, r  , c+1)] = weight_r0c1f0 - MULTIPLIER*delta_w_r0c1f0;
		    (fil_w->data)[ind_fil_w(1, r  , c+1)] = weight_r0c1f1 - MULTIPLIER*delta_w_r0c1f1;
		    (fil_w->data)[ind_fil_w(2, r  , c+1)] = weight_r0c1f2 - MULTIPLIER*delta_w_r0c1f2;

		    (fil_w->data)[ind_fil_w(0, r+1, c  )] = weight_r1c0f0 - MULTIPLIER*delta_w_r1c0f0;
		    (fil_w->data)[ind_fil_w(1, r+1, c  )] = weight_r1c0f1 - MULTIPLIER*delta_w_r1c0f1;
		    (fil_w->data)[ind_fil_w(2, r+1, c  )] = weight_r1c0f2 - MULTIPLIER*delta_w_r1c0f2;

		    (fil_w->data)[ind_fil_w(0, r+1, c+1)] = weight_r1c1f0 - MULTIPLIER*delta_w_r1c1f0;
		    (fil_w->data)[ind_fil_w(1, r+1, c+1)] = weight_r1c1f1 - MULTIPLIER*delta_w_r1c1f1;
		    (fil_w->data)[ind_fil_w(2, r+1, c+1)] = weight_r1c1f2 - MULTIPLIER*delta_w_r1c1f2;
		}

		// for the left over element in the current row, at the end
		for (; c < FIL_COLS; ++c)
		{
		    delta_w0 = 0.0, delta_w1 = 0.0, delta_w2 = 0.0;

		    for (int b = 0; b < BATCH_SIZE; ++b)
		    {
		    	cur_image = shuffle_index[b+base];

    			for (int i = 0; i < N_ROWS_CONV; ++i)
    			{

    				for (int j = 0; j < N_COLS_CONV; ++j)
    				{

    					INCREMENT_FLOPS(9)

    					input_pixel = (input_images->data)[ind_input_img(cur_image, i+r, j+c)];

    					conv_val0 = (conv_t->data)[ind_conv_out(b, 0, i, j)];
    					conv_val1 = (conv_t->data)[ind_conv_out(b, 1, i, j)];
    					conv_val2 = (conv_t->data)[ind_conv_out(b, 2, i, j)];

    					del_conv0 = (del_conv->data)[ind_conv_out(b, 0, i, j)];
    					del_conv1 = (del_conv->data)[ind_conv_out(b, 1, i, j)];
    					del_conv2 = (del_conv->data)[ind_conv_out(b, 2, i, j)];
    					
    					if( conv_val0 > 0.0 )
    					{
    						delta_w0 += del_conv0*input_pixel;
    					}

    					if( conv_val1 > 0.0 )
    					{
    						delta_w1 += del_conv1*input_pixel;
    					}

    					if( conv_val2 > 0.0 )
    					{
    						delta_w2 += del_conv2*input_pixel;
    					}
    				}
    			}
		    }

		    INCREMENT_FLOPS(21)
		    // modifying delta_w to account for sign function applied on weights
		    weight0 = (fil_w->data)[ind_fil_w(0, r, c)];
		    weight1 = (fil_w->data)[ind_fil_w(1, r, c)];
		    weight2 = (fil_w->data)[ind_fil_w(2, r, c)];

		    if (weight0 <= 1 && weight0 >= -1)
		    {
		    	delta_w0 *= ( (alphas[0]*weight0) +  recip_n );
		    }
		    else
	    	{
	    		delta_w0 *= recip_n;
	    	}

	    	if (weight1 <= 1 && weight1 >= -1)
		    {
		    	delta_w1 *= ( (alphas[1]*weight1) +  recip_n );
		    }
		    else
	    	{
	    		delta_w1 *= recip_n;
	    	}

	    	if (weight2 <= 1 && weight2 >= -1)
		    {
		    	delta_w2 *= ( (alphas[2]*weight2) +  recip_n );
		    }
		    else
	    	{
	    		delta_w2 *= recip_n;
	    	}
		    
		    (fil_w->data)[ind_fil_w(0, r, c)] = weight0 - MULTIPLIER*delta_w0;
		    (fil_w->data)[ind_fil_w(1, r, c)] = weight1 - MULTIPLIER*delta_w1;
		    (fil_w->data)[ind_fil_w(2, r, c)] = weight2 - MULTIPLIER*delta_w2;
		}
	}


	// for the leftover row, at the end
	for (; r < FIL_ROWS; ++r)
	{
		for (c=0; c < FIL_COLS; ++c)
		{
		    delta_w0 = 0.0, delta_w1 = 0.0, delta_w2 = 0.0;

		    for (int b = 0; b < BATCH_SIZE; ++b)
		    {
		    	cur_image = shuffle_index[b+base];

    			for (int i = 0; i < N_ROWS_CONV; ++i)
    			{

    				for (int j = 0; j < N_COLS_CONV; ++j)
    				{

    					INCREMENT_FLOPS(9)

    					input_pixel = (input_images->data)[ind_input_img(cur_image, i+r, j+c)];

    					conv_val0 = (conv_t->data)[ind_conv_out(b, 0, i, j)];
    					conv_val1 = (conv_t->data)[ind_conv_out(b, 1, i, j)];
    					conv_val2 = (conv_t->data)[ind_conv_out(b, 2, i, j)];

    					del_conv0 = (del_conv->data)[ind_conv_out(b, 0, i, j)];
    					del_conv1 = (del_conv->data)[ind_conv_out(b, 1, i, j)];
    					del_conv2 = (del_conv->data)[ind_conv_out(b, 2, i, j)];
    					
    					if( conv_val0 > 0.0 )
    					{
    						delta_w0 += del_conv0*input_pixel;
    					}

    					if( conv_val1 > 0.0 )
    					{
    						delta_w1 += del_conv1*input_pixel;
    					}

    					if( conv_val2 > 0.0 )
    					{
    						delta_w2 += del_conv2*input_pixel;
    					}
    				}
    			}
		    }

		    INCREMENT_FLOPS(21)
		    // modifying delta_w to account for sign function applied on weights
		    weight0 = (fil_w->data)[ind_fil_w(0, r, c)];
		    weight1 = (fil_w->data)[ind_fil_w(1, r, c)];
		    weight2 = (fil_w->data)[ind_fil_w(2, r, c)];

		    if (weight0 <= 1 && weight0 >= -1)
		    {
		    	delta_w0 *= ( (alphas[0]*weight0) +  recip_n );
		    }
		    else
	    	{
	    		delta_w0 *= recip_n;
	    	}

	    	if (weight1 <= 1 && weight1 >= -1)
		    {
		    	delta_w1 *= ( (alphas[1]*weight1) +  recip_n );
		    }
		    else
	    	{
	    		delta_w1 *= recip_n;
	    	}

	    	if (weight2 <= 1 && weight2 >= -1)
		    {
		    	delta_w2 *= ( (alphas[2]*weight2) +  recip_n );
		    }
		    else
	    	{
	    		delta_w2 *= recip_n;
	    	}
		    
		    (fil_w->data)[ind_fil_w(0, r, c)] = weight0 - MULTIPLIER*delta_w0;
		    (fil_w->data)[ind_fil_w(1, r, c)] = weight1 - MULTIPLIER*delta_w1;
		    (fil_w->data)[ind_fil_w(2, r, c)] = weight2 - MULTIPLIER*delta_w2;
		}
	}
}

// No unrolling
void update_conv_biases(tensor* fil_b, tensor* del_conv, tensor* conv_t)
{
	double delta_b;
	double conv_val;
	double del_conv_val;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		delta_b = 0.0;

		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_COLS_CONV; j=j+4)
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

		INCREMENT_FLOPS(2)
		(fil_b->data)[f] -= MULTIPLIER*delta_b;
	}
}
