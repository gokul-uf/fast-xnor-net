#include "back_prop.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int N_ROWS_POOL;
int N_COLS_POOL;

void update_sotmax_weights(tensor* fully_con_w, tensor softmax_out, tensor pool_t, int* labels, int base, int shuffle_index[]){

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
						delta = softmax_out.data[offset(&softmax_out, b, 0, 0, d)] - (labels[shuffle_index[base+b]] == d);
						delta_w += delta * pool_t.data[offset(&pool_t, b, c, r, f)];
					}

					(fully_con_w->data)[offset(fully_con_w, d, c, r, f)] -= (LEARN_RATE/BATCH_SIZE)*delta_w;
				}
			}
		}
	}
}

void update_sotmax_biases(tensor* fully_con_b, tensor softmax_out, int* labels, int base, int shuffle_index[]){

	for (int d = 0; d < N_DIGS; ++d)
	{
		double delta_b = 0.0, delta = 0.0;
		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			delta = softmax_out.data[offset(&softmax_out, b, 0, 0, d)] - (labels[shuffle_index[base+b]] == d);
			delta_b += delta;
		}

		(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, d)] -= (LEARN_RATE/BATCH_SIZE)*delta_b;
				
	}
}

void bp_softmax_to_maxpool(tensor* del_max_pool, tensor softmax_out, int* labels, int base, tensor fully_con_w, int shuffle_index[]){
	double sum = 0.0;
	double delta = 0.0;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					sum = 0.0;

					for (int d = 0; d < N_DIGS; ++d)
					{
						delta = softmax_out.data[offset(&softmax_out, b, 0, 0, d)] - (labels[shuffle_index[base+b]] == d);
						sum += delta * fully_con_w.data[offset(&fully_con_w, d, c, r, f)];
					}

					(del_max_pool->data)[offset(del_max_pool, b, c, r, f)] = sum;
				}
			}
		}
	}
}

void bp_maxpool_to_conv(tensor* del_conv, tensor del_max_pool, tensor conv_t, int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL],
   	int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL]){

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					int row = pool_index_i[b][f][r][c];
					int col = pool_index_j[b][f][r][c];

					if (conv_t.data[offset(&conv_t, b, row, col, f)] > 0.0)
					{
						(del_conv->data)[offset(del_conv, b, col, row, f)] = (del_max_pool.data)[offset(&del_max_pool, b, c, r, f)];	
					}
				}
			}
		}
	}
}

void update_conv_weights(double fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], tensor del_conv, tensor conv_t, 
	tensor input_images, int base, int shuffle_index[]){

	for (int f = 0; f < NUM_FILS; ++f)
	{
		for (int r = 0; r < FIL_ROWS; ++r)
		{
			for (int c = 0; c < FIL_COLS; ++c)
			{

				double delta_w = 0.0;

				for (int b = 0; b < BATCH_SIZE; ++b)
				{
					for (int i = 0; i < N_ROWS_CONV; ++i)
					{
						for (int j = 0; j < N_COLS_CONV; ++j)
						{
							if(conv_t.data[offset(&conv_t, b, j, i, f)] > 0.0){
								delta_w += del_conv.data[offset(&del_conv, b, j, i, f)] 
										* input_images.data[offset(&input_images, shuffle_index[b+base], j+c, i+r, 0)];
							}
						}
					}
				}

				fil_w[f][r][c] -= (LEARN_RATE/BATCH_SIZE)*delta_w;
			}
		}
	}
}

void update_conv_biases(double fil_b[NUM_FILS], tensor del_conv, tensor conv_t){

	for (int f = 0; f < NUM_FILS; ++f)
	{
		double delta_b = 0.0;

		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_COLS_CONV; ++j)
				{
					if(conv_t.data[offset(&conv_t, b, j, i, f)] > 0.0){
						delta_b	+= del_conv.data[offset(&del_conv, b, j, i, f)];
					}
				}
			}
		}

		fil_b[f] -= (LEARN_RATE/BATCH_SIZE)*delta_b;
	}
}