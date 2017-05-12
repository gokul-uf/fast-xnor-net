#include "back_prop.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int N_ROWS_POOL;
int N_COLS_POOL;

void update_sotmax_weights(tensor* fully_con_w, tensor* softmax_out, tensor* pool_t, int* labels, int base, int shuffle_index[])
{
	double multiplier = (LEARN_RATE/BATCH_SIZE);
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
						delta = (softmax_out->data)[offset(softmax_out, b, 0, 0, d)] - (labels[shuffle_index[base+b]] == d);
						delta_w += delta * (pool_t->data)[offset(pool_t, b, c, r, f)];
					}

					(fully_con_w->data)[offset(fully_con_w, d, c, r, f)] -= multiplier*delta_w;
				}
			}
		}
	}
}

void update_sotmax_biases(tensor* fully_con_b, tensor* softmax_out, int* labels, int base, int shuffle_index[])
{
	double multiplier = (LEARN_RATE/BATCH_SIZE);
	double delta0=0, delta1=0, delta2=0, delta3=0, delta4=0, delta5=0, delta6=0, delta7=0, delta8=0, delta9=0;
	double delta00=0, delta11=0, delta22=0, delta33=0, delta44=0, delta55=0, delta66=0, delta77=0, delta88=0, delta99=0;

	int true_label1, true_label2;

	for (int b = 0; b < BATCH_SIZE; b=b+2)
	{
		true_label1 = labels[shuffle_index[base+b]];

		delta0 += softmax_out->data[offset(softmax_out, b, 0, 0, 0)] - (true_label1 == 0);
		delta1 += softmax_out->data[offset(softmax_out, b, 0, 0, 1)] - (true_label1 == 1);
		delta2 += softmax_out->data[offset(softmax_out, b, 0, 0, 2)] - (true_label1 == 2);
		delta3 += softmax_out->data[offset(softmax_out, b, 0, 0, 3)] - (true_label1 == 3);
		delta4 += softmax_out->data[offset(softmax_out, b, 0, 0, 4)] - (true_label1 == 4);
		delta5 += softmax_out->data[offset(softmax_out, b, 0, 0, 5)] - (true_label1 == 5);
		delta6 += softmax_out->data[offset(softmax_out, b, 0, 0, 6)] - (true_label1 == 6);
		delta7 += softmax_out->data[offset(softmax_out, b, 0, 0, 7)] - (true_label1 == 7);
		delta8 += softmax_out->data[offset(softmax_out, b, 0, 0, 8)] - (true_label1 == 8);
		delta9 += softmax_out->data[offset(softmax_out, b, 0, 0, 9)] - (true_label1 == 9);


		true_label2 = labels[shuffle_index[base+b+1]];

		delta00 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 0)] - (true_label2 == 0);
		delta11 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 1)] - (true_label2 == 1);
		delta22 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 2)] - (true_label2 == 2);
		delta33 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 3)] - (true_label2 == 3);
		delta44 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 4)] - (true_label2 == 4);
		delta55 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 5)] - (true_label2 == 5);
		delta66 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 6)] - (true_label2 == 6);
		delta77 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 7)] - (true_label2 == 7);
		delta88 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 8)] - (true_label2 == 8);
		delta99 += softmax_out->data[offset(softmax_out, b+1, 0, 0, 9)] - (true_label2 == 9);

	}

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

	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 0)] -= multiplier*delta0;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 1)] -= multiplier*delta1;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 2)] -= multiplier*delta2;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 3)] -= multiplier*delta3;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 4)] -= multiplier*delta4;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 5)] -= multiplier*delta5;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 6)] -= multiplier*delta6;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 7)] -= multiplier*delta7;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 8)] -= multiplier*delta8;
	(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, 9)] -= multiplier*delta9;
}

void bp_softmax_to_maxpool(tensor* del_max_pool, tensor* softmax_out, int* labels, int base, tensor* fully_con_w, int shuffle_index[])
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
}

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

void bp_softmax_to_conv(tensor* del_conv, tensor* softmax_out, tensor* conv_t, int* labels, int base, tensor* fully_con_w, 
	int shuffle_index[], int pool_index_i[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[BATCH_SIZE][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
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


					// bp from max_pool to conv
					int row = pool_index_i[b][f][r][c];
					int col = pool_index_j[b][f][r][c];

					if (conv_t->data[offset(conv_t, b, row, col, f)] > 0.0)
					{
						(del_conv->data)[offset(del_conv, b, col, row, f)] = sum_fin;
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

void bin_update_conv_weights(tensor* fil_w, tensor* fil_bin_w, double alphas[NUM_FILS], tensor* del_conv, tensor* conv_t, 
								tensor* input_images, int base, int shuffle_index[])
{
	double recip_n = 1.0/(FIL_ROWS * FIL_COLS);
	double multiplier = (LEARN_RATE/BATCH_SIZE);
	double delta_w;
	int cur_image;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		for (int r = 0; r < FIL_ROWS; ++r)
		{
			for (int c = 0; c < FIL_COLS; ++c)
			{
                for(int d = 0; d < FIL_DEPTH; ++d)
                {
				    delta_w = 0.0;

				    for (int b = 0; b < BATCH_SIZE; ++b)
				    {
				    	cur_image = shuffle_index[b+base];
		    			for (int i = 0; i < N_ROWS_CONV; ++i)
		    			{
		    				for (int j = 0; j < N_COLS_CONV; ++j)
		    				{
		    					if( (conv_t->data)[offset(conv_t, b, j  , i, f)] > 0.0 )
		    					{
		    						delta_w += (del_conv->data)[offset(del_conv, b, j, i, f)]*
		    								(input_images->data)[offset(input_images, cur_image, j+c, i+r, 0)];
		    					}
		    				}
		    			}
				    }

				    // modifying delta_w to account for sign function applied on weights
				    double weight = (fil_w->data)[offset(fil_w, f, r, c, d)];
				    if (weight <= 1 && weight >= -1)
				    {
				    	delta_w *= ( (alphas[f]*weight) +  recip_n);
				    }
				    else
			    	{
			    		delta_w *= recip_n;
			    	}
				    
				    (fil_w->data)[offset(fil_w, f, r, c, d)] = weight - multiplier*delta_w;
                }
			}
		}
	}
}

void update_conv_biases(tensor* fil_b, tensor* del_conv, tensor* conv_t)
{
	double multiplier = (LEARN_RATE/BATCH_SIZE);
	double delta_b;

	for (int f = 0; f < NUM_FILS; ++f)
	{
		delta_b = 0.0;

		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_COLS_CONV; j=j+4)
				{
					if(conv_t->data[offset(conv_t, b, j, i, f)] > 0.0){
						delta_b	+= del_conv->data[offset(del_conv, b, j, i, f)];
					}
				}
			}
		}

		(fil_b->data)[offset(fil_b, f, 0, 0, 0)] -= multiplier*delta_b;
	}
}
