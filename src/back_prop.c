#include "back_prop.h"

void update_sotmax_weights(tensor* fully_con_w, tensor* fully_con_b, tensor softmax_out, tensor pool_t, int* labels, int base){

	for (int d = 0; d < N_DIGS; ++d)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int r = 0; r < N_ROWS_POOL; ++r)
			{
				for (int c = 0; c < N_COLS_POOL; ++c)
				{
					double delta_w = 0.0, delta_b = 0.0, delta = 0.0;
					for (int b = 0; b < BATCH_SIZE; ++b)
					{
						delta = softmax_out.data[offset(&softmax_out, b, c, r, f)] - labels[base+b];
						delta_w += delta * pool_t.data[offset(&pool_t, b, c, r, f)];

						delta_b += delta;
					}

					(fully_con_w->data)[offset(fully_con_w, d, c, r, f)] -= (LEARN_RATE/BATCH_SIZE)*delta_w;
					(fully_con_b->data)[offset(fully_con_b, 0, 0, 0, d)] -= (LEARN_RATE/BATCH_SIZE)*delta_b;
				}
			}
		}
	}
}