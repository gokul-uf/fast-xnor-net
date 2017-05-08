#include "back_prop.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int N_ROWS_POOL;
int N_COLS_POOL;

void update_sotmax_weights(tensor* fully_con_w, tensor* softmax_out, tensor* pool_t, int* labels, int base, int shuffle_index[]){

	for (int d = 0; d < N_DIGS; ++d)
	{
		COST_INC_I_ADD(1); // d++
		for (int f = 0; f < NUM_FILS; ++f)
		{
			COST_INC_I_ADD(1); // f++
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				COST_INC_I_ADD(1); // r++
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					COST_INC_I_ADD(1); // c++
					double delta_w = 0.0, delta = 0.0;
					for (int b = 0; b < BATCH_SIZE; ++b)
					{
						COST_INC_I_ADD(1); // b++

						COST_INC_F_ADD(2); COST_INC_I_ADD(1); COST_INC_F_MUL(1);
						delta = softmax_out->data[offset(softmax_out, b, 0, 0, d)] - (labels[shuffle_index[base+b]] == d);
						delta_w += delta * pool_t->data[offset(pool_t, b, c, r, f)];
					}

					COST_INC_F_ADD(1); COST_INC_F_MUL(1); COST_INC_F_DIV(1);
					(fully_con_w->data)[offset(fully_con_w, d, c, r, f)] -= (LEARN_RATE/BATCH_SIZE)*delta_w;
				}
			}
		}
	}
}

void update_sotmax_biases(tensor* fully_con_b, tensor* softmax_out, int* labels, int base, int shuffle_index[]){

	for (int d = 0; d < N_DIGS; ++d)
	{
		COST_INC_I_ADD(1); // d++
		double delta_b = 0.0, delta = 0.0;
		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			COST_INC_I_ADD(1); // b++

			COST_INC_F_ADD(2); COST_INC_I_ADD(1);
			delta = softmax_out->data[offset(softmax_out, b, 0, 0, d)] - (labels[shuffle_index[base+b]] == d);
			delta_b += delta;
		}

		COST_INC_F_ADD(1); COST_INC_F_MUL(1); COST_INC_F_DIV(1);
		(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, d)] -= (LEARN_RATE/BATCH_SIZE)*delta_b;

	}
}

void bp_softmax_to_maxpool(tensor* del_max_pool, tensor* softmax_out, int* labels, int base, tensor* fully_con_w,
	int shuffle_index[]){
	double sum = 0.0;
	double delta = 0.0;

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		COST_INC_I_ADD(1); // b++
		for (int f = 0; f < NUM_FILS; ++f)
		{
			COST_INC_I_ADD(1); // f++
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				COST_INC_I_ADD(1); // r++
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					COST_INC_I_ADD(1); // c++
					sum = 0.0;

					for (int d = 0; d < N_DIGS; ++d)
					{
						COST_INC_I_ADD(1); // d++

						COST_INC_F_ADD(1); COST_INC_I_ADD(1);
						delta = (softmax_out->data)[offset(softmax_out, b, 0, 0, d)] - (labels[shuffle_index[base+b]] == d);

						COST_INC_F_ADD(1); COST_INC_F_MUL(1);
						sum += delta * (fully_con_w->data)[offset(fully_con_w, d, c, r, f)];
					}

					(del_max_pool->data)[offset(del_max_pool, b, c, r, f)] = sum;

				}
			}
		}
	}
}

void bp_maxpool_to_conv(tensor* del_conv, tensor* del_max_pool, tensor* conv_t, int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL],
   	int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL]){

	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		COST_INC_I_ADD(1); //b++
		for (int f = 0; f < NUM_FILS; ++f)
		{
			COST_INC_I_ADD(1); //f++
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				COST_INC_I_ADD(1); //r++
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					COST_INC_I_ADD(1); //c++
					int row = pool_index_i[b][f][r][c];
					int col = pool_index_j[b][f][r][c];

					if (conv_t->data[offset(conv_t, b, row, col, f)] > 0.0)
					{
						(del_conv->data)[offset(del_conv, b, col, row, f)] = (del_max_pool->data)[offset(del_max_pool, b, c, r, f)];
					}
				}
			}
		}
	}
}

void update_conv_weights(tensor* fil_w, tensor* del_conv, tensor* conv_t, tensor* input_images,
							int base, int shuffle_index[]){

	for (int f = 0; f < NUM_FILS; ++f)
	{
		COST_INC_I_ADD(1); //f++
		for (int r = 0; r < FIL_ROWS; ++r)
		{
			COST_INC_I_ADD(1); //r++
			for (int c = 0; c < FIL_COLS; ++c)
			{
				COST_INC_I_ADD(1); //c++
        for(int d = 0; d < FIL_DEPTH; ++d){
					COST_INC_I_ADD(1); //d++
				    double delta_w = 0.0;

				    for (int b = 0; b < BATCH_SIZE; ++b)
				    {
							COST_INC_I_ADD(1); //b++
				    	for (int i = 0; i < N_ROWS_CONV; ++i)
				    	{
								COST_INC_I_ADD(1); //i++
				    		for (int j = 0; j < N_COLS_CONV; ++j)
				    		{
									COST_INC_I_ADD(1); //j++
				    			if( (conv_t->data)[offset(conv_t, b, j, i, f)] > 0.0 ){
										COST_INC_F_ADD(1); COST_INC_F_MUL(1); COST_INC_I_ADD(3);
				    				delta_w += (del_conv->data)[offset(del_conv, b, j, i, f)]
				    						* (input_images->data)[offset(input_images, shuffle_index[b+base], j+c, i+r, d)];
				    			}
				    		}
				    	}
				    }
						COST_INC_F_ADD(1); COST_INC_F_DIV(1); COST_INC_F_MUL(1);
				    (fil_w->data)[offset(fil_w, f, r, c, d)] -= (LEARN_RATE/BATCH_SIZE)*delta_w;
        }
			}
		}
	}
}

void update_conv_biases(tensor* fil_b, tensor* del_conv, tensor* conv_t){

	for (int f = 0; f < NUM_FILS; ++f)
	{
		COST_INC_I_ADD(1); //f++
		double delta_b = 0.0;

		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			COST_INC_I_ADD(1); //b++
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				COST_INC_I_ADD(1); //++i
				for (int j = 0; j < N_COLS_CONV; ++j)
				{
					COST_INC_I_ADD(1); //j++
					if(conv_t->data[offset(conv_t, b, j, i, f)] > 0.0){
						COST_INC_F_ADD(1);
						delta_b	+= del_conv->data[offset(del_conv, b, j, i, f)];
					}
				}
			}
		}
		COST_INC_F_ADD(1); COST_INC_F_DIV(1); COST_INC_F_MUL(1);
		(fil_b->data)[offset(fil_b, f, 0, 0, 0)] -= (LEARN_RATE/BATCH_SIZE)*delta_b;
	}
}
