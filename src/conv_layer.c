#include "conv_layer.h"

int N_ROWS_CONV;
int N_COLS_CONV;
int TOTAL_FLOPS;

void print_register_d(__m256d reg){
    double temp[4];
    _mm256_store_pd(temp, reg);
    printf("%.2f %.2f %.2f %.2f\n", temp[0] ,temp[1] , temp[2] , temp[3]);
}
void print_register_f(__m256 reg){
    __m256 temp = _mm256_set_ps(1,2,3,4,5,6,7,8);
    printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", temp[0] ,temp[1] , temp[2] , temp[3], temp[4] ,temp[5] , temp[6] , temp[7]);
}
void convolution(tensor* input_t, tensor* conv_t, int batch_size,
	tensor* fil_w, tensor* fil_b, int base, int shuffle_index[])
{
	for (int b = 0; b < batch_size; ++b)
	{
		for (int f = 0; f < NUM_FILS; ++f)
		{
			for (int i = 0; i < N_ROWS_CONV; ++i)
			{
				for (int j = 0; j < N_ROWS_CONV; ++j)
				{
					(conv_t->data)[offset(conv_t,b,j,i,f)] = convolve(input_t, i, j, b+base, fil_w, fil_b, f, shuffle_index);
				}
			}
		}
	}
}

// Tested!!
double convolve(tensor* t, int r, int c, int image_num, tensor* fil_w, tensor* fil_b, int f, int shuffle_index[])
{

	double conv_val = 0.0;
	for (int i = 0; i < FIL_ROWS; ++i)
	{
		for (int j = 0; j < FIL_COLS; ++j)
		{
            for (int k = 0; k < FIL_DEPTH; ++k){

            	INCREMENT_FLOPS(2)

			    conv_val += (fil_w->data)[offset(fil_w, f, j, i, k)] * (t->data)[offset(t, shuffle_index[image_num], c+j, r+i, k)];
            }
		}
	}

	INCREMENT_FLOPS(2)

	conv_val += (fil_b->data)[offset(fil_b, f, 0, 0, 0)];

	// applying ReLU
	if (conv_val < 0.0)
	{
		conv_val = 0.0;
	}

	return conv_val;
}

// convolution and pooling together, Ben's changes
void bin_convolve_pool(tensor* input_t, tensor* conv_t, tensor* pool_t, int batch_size,
	int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS], tensor fil_b, int base, int shuffle_index[],
	int pool_index_i[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{

	double conv_val_r0c0_f0, conv_val_r0c0_f1, conv_val_r0c0_f2;
	double conv_val_r0c1_f0, conv_val_r0c1_f1, conv_val_r0c1_f2;
	double conv_val_r1c0_f0, conv_val_r1c0_f1, conv_val_r1c0_f2;
	double conv_val_r1c1_f0, conv_val_r1c1_f1, conv_val_r1c1_f2;

	__m256d m_conv_val_f0, m_conv_val_f1, m_conv_val_f2;
	__m256d m_alpha_f0 = _mm256_set1_pd(alphas[0]);
	__m256d m_alpha_f1 = _mm256_set1_pd(alphas[1]);
	__m256d m_alpha_f2 = _mm256_set1_pd(alphas[2]);

	__m256d m_bias_f0 = _mm256_set1_pd(fil_b.data[0]);
	__m256d m_bias_f1 = _mm256_set1_pd(fil_b.data[1]);
	__m256d m_bias_f2 = _mm256_set1_pd(fil_b.data[2]);


	__m256d zeroes_p = _mm256_set1_pd( 0.0);
	__m256d ones_p   = _mm256_set1_pd( 1.0);
	__m256d mones_p  = _mm256_set1_pd(-1.0);

	__m256d m_input_pixel;

	int cur_image;

	double max_1_f0,   max_2_f0,     max_f0;
	int  max_1_i_f0, max_1_j_f0, max_2_i_f0, max_2_j_f0, max_i_f0, max_j_f0;
	double max_1_f1,   max_2_f1,     max_f1;
	int  max_1_i_f1, max_1_j_f1, max_2_i_f1, max_2_j_f1, max_i_f1, max_j_f1;
	double max_1_f2,   max_2_f2,    max_f2;
	int  max_1_i_f2, max_1_j_f2, max_2_i_f2, max_2_j_f2, max_i_f2, max_j_f2;
	int pool_i, pool_j;
	double prev1, prev2;
    int conv_f_index100 = N_ROWS_CONV * N_COLS_CONV;
    int conv_f_index000 = 0;
    int conv_f_index001 = 1;
    int conv_f_index110 = conv_f_index100 + N_COLS_CONV;
    int conv_f_index101 = conv_f_index100 + 1;
    int conv_f_index111 = conv_f_index110 + 1;
    int conv_f_index200 = 2 * N_ROWS_CONV * N_COLS_CONV;
    int conv_f_index010 = N_COLS_CONV;
    int conv_f_index011 = N_COLS_CONV + 1;
    int conv_f_index210 = conv_f_index200 + N_COLS_CONV;
    int conv_f_index201 = conv_f_index200 + 1;
    int conv_f_index211 = conv_f_index210 + 1;
    int CONV_SIZE = NUM_FILS * N_ROWS_CONV * N_COLS_CONV;
    int _2_N_COLS_CONV = 2 * N_COLS_CONV;
    int pool_f_index0 = 0;
    int pool_f_index1 = N_ROWS_POOL * N_COLS_POOL;
    int pool_f_index2 = N_ROWS_POOL * N_COLS_POOL * 2;
    int POOL_SIZE = NUM_FILS * N_ROWS_POOL * N_COLS_POOL;
    double* conv_data = conv_t->data;
    double* input_data = input_t->data;
    double* pool_data = pool_t->data;

	for (int b = 0; b < batch_size; ++b)
	{
		cur_image = shuffle_index[b+base];
        int image_r_index0 = cur_image * IMAGE_ROWS * IMAGE_COLS;
        int image_r_index1 = image_r_index0 + IMAGE_COLS;
        int conv_r_index000 = conv_f_index000;
        int conv_r_index001 = conv_f_index001;
        int conv_r_index010 = conv_f_index010;
        int conv_r_index011 = conv_f_index011;
        int conv_r_index100 = conv_f_index100;
        int conv_r_index101 = conv_f_index101;
        int conv_r_index110 = conv_f_index110;
        int conv_r_index111 = conv_f_index111;
        int conv_r_index200 = conv_f_index200;
        int conv_r_index201 = conv_f_index201;
        int conv_r_index210 = conv_f_index210;
        int conv_r_index211 = conv_f_index211;
        int pool_i_index0 = pool_f_index0;
        int pool_i_index1 = pool_f_index1;
        int pool_i_index2 = pool_f_index2;
		// Unroll r and c loops by 2 so that max pooling can be merged with convolution
		for (int r = 0, pool_i=0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
		{
            int conv_c_index000 = conv_r_index000;
            int conv_c_index001 = conv_r_index001;
            int conv_c_index010 = conv_r_index010;
            int conv_c_index011 = conv_r_index011;
            int conv_c_index100 = conv_r_index100;
            int conv_c_index101 = conv_r_index101;
            int conv_c_index110 = conv_r_index110;
            int conv_c_index111 = conv_r_index111;
            int conv_c_index200 = conv_r_index200;
            int conv_c_index201 = conv_r_index201;
            int conv_c_index210 = conv_r_index210;
            int conv_c_index211 = conv_r_index211;
            int pool_j_index0 = pool_i_index0;
            int pool_j_index1 = pool_i_index1;
            int pool_j_index2 = pool_i_index2;
			for (int c = 0, pool_j=0; c+1 < N_ROWS_CONV; c=c+2, ++pool_j)
			{

				m_conv_val_f0 = _mm256_set1_pd(0.0);
				m_conv_val_f1 = _mm256_set1_pd(0.0);
				m_conv_val_f2 = _mm256_set1_pd(0.0);

        int image_c_index0 = image_r_index0 + c;
        int image_c_index1 = image_r_index1 + c;
				for (int i = 0; i < FIL_ROWS; ++i)
				{
					prev1 = input_data[image_c_index0];
					prev2 = input_data[image_c_index1];
          int image_j_index0 = image_c_index0+1;
          int image_j_index1 = image_c_index1+1;
					for (int j = 0; j < FIL_COLS; ++j)
					{
						m_input_pixel = _mm256_set_pd(input_data[image_j_index1], prev2, input_data[image_j_index0], prev1);
						
						INCREMENT_FLOPS(12)
						// --------------------------------------------filter 0-------------------------------------
						if (fil_bin_w[0][i][j] == 1)
						{
							m_conv_val_f0 = _mm256_add_pd(m_conv_val_f0, m_input_pixel);
						}
						else
						{
							m_conv_val_f0 = _mm256_sub_pd(m_conv_val_f0, m_input_pixel);
						}
						// --------------------------------------------filter 1-----------------------------------
						if (fil_bin_w[1][i][j] == 1)
						{
							m_conv_val_f1 = _mm256_add_pd(m_conv_val_f1, m_input_pixel);
						}
						else
						{
							m_conv_val_f1 = _mm256_sub_pd(m_conv_val_f1, m_input_pixel);
						}
						// -------------------------------------------filter 2----------------------------------------------
						if (fil_bin_w[2][i][j] == 1)
						{
							m_conv_val_f2 = _mm256_add_pd(m_conv_val_f2, m_input_pixel);
						}
						else
						{
							m_conv_val_f2 = _mm256_add_pd(m_conv_val_f2, m_input_pixel);
						}

						prev1 = input_data[image_j_index0];
						prev2 = input_data[image_j_index1];
            image_j_index0 += 1;
            image_j_index1 += 1;
					}
          image_c_index0 += IMAGE_COLS;
          image_c_index1 += IMAGE_COLS;
				}
				INCREMENT_FLOPS(48)
				// -----------------------------------------------filter 0 ----------------------------------------------
				m_conv_val_f0 = _mm256_mul_pd(m_conv_val_f0, m_alpha_f0);
				m_conv_val_f0 = _mm256_add_pd(m_conv_val_f0, m_bias_f0);

				// -----------------------------------------------filter 1---------------------------------------------

				m_conv_val_f1 = _mm256_mul_pd(m_conv_val_f1, m_alpha_f1);
				m_conv_val_f1 = _mm256_add_pd(m_conv_val_f1, m_bias_f1);				// -----------------------------------------------filter 2---------------------------------------------

				m_conv_val_f2 = _mm256_mul_pd(m_conv_val_f2, m_alpha_f2);
				m_conv_val_f2 = _mm256_add_pd(m_conv_val_f2, m_bias_f2);
				// applying ReLU
				// -------------------------------------------filter 0------------------------------------------------

				m_conv_val_f0 = _mm256_and_pd(m_conv_val_f0, _mm256_cmp_pd(m_conv_val_f0, zeroes_p, 0x0d));
				// -------------------------------------------filter 1------------------------------------------------

				m_conv_val_f1 = _mm256_and_pd(m_conv_val_f1, _mm256_cmp_pd(m_conv_val_f1, zeroes_p, 0x0d));
				// -------------------------------------------filter 2------------------------------------------------

				m_conv_val_f2 = _mm256_and_pd(m_conv_val_f2, _mm256_cmp_pd(m_conv_val_f2, zeroes_p, 0x0d));

				double tmp_f0[4], tmp_f1[4], tmp_f2[4];
      	_mm256_store_pd(tmp_f0, m_conv_val_f0);
				_mm256_store_pd(tmp_f1, m_conv_val_f1);
				_mm256_store_pd(tmp_f2, m_conv_val_f2);

				conv_val_r0c0_f0 = tmp_f0[0];
				conv_val_r0c1_f0 = tmp_f0[1];
				conv_val_r1c0_f0 = tmp_f0[2];
				conv_val_r1c1_f0 = tmp_f0[3];
				conv_val_r0c0_f1 = tmp_f1[0];
				conv_val_r0c1_f1 = tmp_f1[1];
				conv_val_r1c0_f1 = tmp_f1[2];
				conv_val_r1c1_f1 = tmp_f1[3];
				conv_val_r0c0_f2 = tmp_f2[0];
				conv_val_r0c1_f2 = tmp_f2[1];
				conv_val_r1c0_f2 = tmp_f2[2];
				conv_val_r1c1_f2 = tmp_f2[3];

				conv_data[conv_c_index000] = conv_val_r0c0_f0;
				conv_data[conv_c_index001] = conv_val_r0c1_f0;
				conv_data[conv_c_index010] = conv_val_r1c0_f0;
				conv_data[conv_c_index011] = conv_val_r1c1_f0;
				conv_data[conv_c_index100] = conv_val_r0c0_f1;
				conv_data[conv_c_index101] = conv_val_r0c1_f1;
				conv_data[conv_c_index110] = conv_val_r1c0_f1;
				conv_data[conv_c_index111] = conv_val_r1c1_f1;
				conv_data[conv_c_index200] = conv_val_r0c0_f2;
				conv_data[conv_c_index201] = conv_val_r0c1_f2;
				conv_data[conv_c_index210] = conv_val_r1c0_f2;
				conv_data[conv_c_index211] = conv_val_r1c1_f2;

				// --------------------------------------------Max Pooling-------------------------------------
				INCREMENT_FLOPS(9)
				// -------------------------------------------Filter 0----------------------------------------
				if (conv_val_r0c0_f0 > conv_val_r0c1_f0)
				{
					max_1_f0 = conv_val_r0c0_f0;
					max_1_i_f0 = r;
					max_1_j_f0 = c;
				}
				else
				{
					  max_1_f0 = conv_val_r0c1_f0;
					max_1_i_f0 = r;
					max_1_j_f0 = c+1;
				}
				if (conv_val_r1c0_f0 > conv_val_r1c1_f0)
				{
					  max_2_f0 = conv_val_r1c0_f0;
					max_2_i_f0 = r+1;
					max_2_j_f0 = c;
				}
				else
				{
					  max_2_f0 = conv_val_r1c1_f0;
					max_2_i_f0 = r+1;
					max_2_j_f0 = c+1;
				}
				if (max_1_f0 > max_2_f0)
				{
					  max_f0 =   max_1_f0;
					max_i_f0 = max_1_i_f0;
					max_j_f0 = max_1_j_f0;
				}
				else
				{
					  max_f0 =   max_2_f0;
					max_i_f0 = max_2_i_f0;
					max_j_f0 = max_2_j_f0;
				}
				pool_data[pool_j_index0] = max_f0;
				pool_index_i[b][0][pool_i][pool_j] = max_i_f0;
				pool_index_j[b][0][pool_i][pool_j] = max_j_f0;
				// -------------------------------------------Filter 1----------------------------------------
				if (conv_val_r0c0_f1 > conv_val_r0c1_f1)
				{
					max_1_f1 = conv_val_r0c0_f1;
					max_1_i_f1 = r;
					max_1_j_f1 = c;
				}
				else
				{
					  max_1_f1 = conv_val_r0c1_f1;
					max_1_i_f1 = r;
					max_1_j_f1 = c+1;
				}
				if (conv_val_r1c0_f1 > conv_val_r1c1_f1)
				{
					  max_2_f1 = conv_val_r1c0_f1;
					max_2_i_f1 = r+1;
					max_2_j_f1 = c;
				}
				else
				{
					  max_2_f1 = conv_val_r1c1_f1;
					max_2_i_f1 = r+1;
					max_2_j_f1 = c+1;
				}
				if (max_1_f1 > max_2_f1)
				{
					  max_f1 =   max_1_f1;
					max_i_f1 = max_1_i_f1;
					max_j_f1 = max_1_j_f1;
				}
				else
				{
					  max_f1 =   max_2_f1;
					max_i_f1 = max_2_i_f1;
					max_j_f1 = max_2_j_f1;
				}
				pool_data[pool_j_index1] = max_f1;
				pool_index_i[b][1][pool_i][pool_j] = max_i_f1;
				pool_index_j[b][1][pool_i][pool_j] = max_j_f1;
				// -------------------------------------------Filter 2----------------------------------------
				if (conv_val_r0c0_f2 > conv_val_r0c1_f2)
				{
					  max_1_f2 = conv_val_r0c0_f2;
					max_1_i_f2 = r;
					max_1_j_f2 = c;
				}
				else
				{
					  max_1_f2 = conv_val_r0c1_f2;
					max_1_i_f2 = r;
					max_1_j_f2 = c+1;
				}
				if (conv_val_r1c0_f2 > conv_val_r1c1_f2)
				{
					  max_2_f2 = conv_val_r1c0_f2;
					max_2_i_f2 = r+1;
					max_2_j_f2 = c;
				}
				else
				{
					  max_2_f2 = conv_val_r1c1_f2;
					max_2_i_f2 = r+1;
					max_2_j_f2 = c+1;
				}
				if (max_1_f2 > max_2_f2)
				{
					  max_f2 =   max_1_f2;
					max_i_f2 = max_1_i_f2;
					max_j_f2 = max_1_j_f2;
				}
				else
				{
					  max_f2 =   max_2_f2;
					max_i_f2 = max_2_i_f2;
					max_j_f2 = max_2_j_f2;
				}
				pool_data[pool_j_index2] = max_f2;
				pool_index_i[b][2][pool_i][pool_j] = max_i_f2;
				pool_index_j[b][2][pool_i][pool_j] = max_j_f2;
        conv_c_index000 += 2;
        conv_c_index001 += 2;
        conv_c_index010 += 2;
        conv_c_index011 += 2;
        conv_c_index100 += 2;
        conv_c_index101 += 2;
        conv_c_index110 += 2;
        conv_c_index111 += 2;
        conv_c_index200 += 2;
        conv_c_index201 += 2;
        conv_c_index210 += 2;
        conv_c_index211 += 2;
        pool_j_index0 += 1;
        pool_j_index1 += 1;
        pool_j_index2 += 1;
			}
      image_r_index0 += 2 * IMAGE_COLS;
      image_r_index1 = image_r_index0 + IMAGE_COLS;
      conv_r_index000 += _2_N_COLS_CONV;
      conv_r_index001 += _2_N_COLS_CONV;
      conv_r_index010 += _2_N_COLS_CONV;
      conv_r_index011 += _2_N_COLS_CONV;
      conv_r_index100 += _2_N_COLS_CONV;
      conv_r_index101 += _2_N_COLS_CONV;
      conv_r_index110 += _2_N_COLS_CONV;
      conv_r_index111 += _2_N_COLS_CONV;
      conv_r_index200 += _2_N_COLS_CONV;
      conv_r_index201 += _2_N_COLS_CONV;
      conv_r_index210 += _2_N_COLS_CONV;
      conv_r_index211 += _2_N_COLS_CONV;
      pool_i_index0 += N_COLS_POOL;
      pool_i_index1 += N_COLS_POOL;
      pool_i_index2 += N_COLS_POOL;
		}
    conv_f_index000 += CONV_SIZE;
    conv_f_index001 += CONV_SIZE;
    conv_f_index010 += CONV_SIZE;
    conv_f_index011 += CONV_SIZE;
    conv_f_index100 += CONV_SIZE;
    conv_f_index101 += CONV_SIZE;
    conv_f_index110 += CONV_SIZE;
    conv_f_index111 += CONV_SIZE;
    conv_f_index200 += CONV_SIZE;
    conv_f_index201 += CONV_SIZE;
    conv_f_index210 += CONV_SIZE;
    conv_f_index211 += CONV_SIZE;
    pool_f_index0 += POOL_SIZE;
    pool_f_index1 += POOL_SIZE;
    pool_f_index2 += POOL_SIZE;
	}
}

// loop on conv rows and cols unrolled by 2, max-pooling done, pool indexes saved; loop on number of filters unrolled
void xnor_convolve_pool(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV],
					tensor* conv_t, int batch_size, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS],
					double alphas[NUM_FILS], tensor fil_b, tensor* pool_t,
					int pool_index_i[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL], int pool_index_j[][NUM_FILS][N_ROWS_POOL][N_COLS_POOL])
{
	double prev1, prev2, curr1, curr2;

	int pool_i, pool_j;
	int max_pool_1, max_pool_2, max_pool;
	int    ind_1_i,    ind_2_i,    ind_i;
	int    ind_1_j,    ind_2_j,    ind_j;

	int max_pool_1_f0, max_pool_2_f0, max_pool_f0;
	int    ind_1_i_f0,    ind_2_i_f0,    ind_i_f0;
	int    ind_1_j_f0,    ind_2_j_f0,    ind_j_f0;

	int max_pool_1_f1, max_pool_2_f1, max_pool_f1;
	int    ind_1_i_f1,    ind_2_i_f1,    ind_i_f1;
	int    ind_1_j_f1,    ind_2_j_f1,    ind_j_f1;

	int max_pool_1_f2, max_pool_2_f2, max_pool_f2;
	int    ind_1_i_f2,    ind_2_i_f2,    ind_i_f2;
	int    ind_1_j_f2,    ind_2_j_f2,    ind_j_f2;

	__m256d zeroes_p = _mm256_set1_pd( 0.0);
	__m256d ones_p   = _mm256_set1_pd( 1.0);
	__m256d mones_p  = _mm256_set1_pd(-1.0);

	__m256d m_conv_val_f0, m_conv_val_f1, m_conv_val_f2;
	__m256d m_alpha_f0;
	__m256d m_alpha_f1;
	__m256d m_alpha_f2;
	__m256d m_bias_f0;
	__m256d m_bias_f1;
	__m256d m_bias_f2;
	__m256d m_beta;
  __m256d m_weight_f0, m_weight_f1, m_weight_f2 ;
  __m256d m_input_pixel;

  __m256 m_conv_val_ps_f0, m_conv_val_ps_f1, m_conv_val_ps_f2;
  __m256 m_weight_ps_f0, m_weight_ps_f1, m_weight_ps_f2 ;
  __m256 m_input_pixel_ps;

	double conv_val0_f0;
	double conv_val1_f0;
	double conv_val2_f0;
	double conv_val3_f0;
	double conv_val0_f1;
	double conv_val1_f1;
	double conv_val2_f1;
	double conv_val3_f1;
	double conv_val0_f2;
	double conv_val1_f2;
	double conv_val2_f2;
	double conv_val3_f2;

	double weight_f0;
	double weight_f1;
	double weight_f2;

	int f, j00, j10, j01, j11, j02, j12;

  int SINGLE_FIL_CONV_SIZE = N_ROWS_CONV * N_COLS_CONV;

  int conv_b_index100 = N_ROWS_CONV * N_COLS_CONV;
  int conv_b_index000 = 0;
  int conv_b_index001 = 1;
  int conv_b_index110 = conv_b_index100 + N_COLS_CONV;
  int conv_b_index101 = conv_b_index100 + 1;
  int conv_b_index111 = conv_b_index110 + 1;
  int conv_b_index200 = 2 * N_ROWS_CONV * N_COLS_CONV;
  int conv_b_index010 = N_COLS_CONV;
  int conv_b_index011 = N_COLS_CONV + 1;
  int conv_b_index210 = conv_b_index200 + N_COLS_CONV;
  int conv_b_index201 = conv_b_index200 + 1;
  int conv_b_index211 = conv_b_index210 + 1;

  int _2_N_COLS_CONV = 2 * N_COLS_CONV;
  int CONV_SIZE = NUM_FILS * SINGLE_FIL_CONV_SIZE;
  int THREE_FIL_CONV_SIZE = 3 * SINGLE_FIL_CONV_SIZE;

  double* conv_data = conv_t->data;
  double* pool_data = pool_t->data;

	for (int b = 0; b < batch_size; ++b)
	{
    int conv_f_index100 = conv_b_index100;
    int conv_f_index000 = conv_b_index000;
    int conv_f_index001 = conv_b_index001;
    int conv_f_index110 = conv_b_index110;
    int conv_f_index101 = conv_b_index101;
    int conv_f_index111 = conv_b_index111;
    int conv_f_index200 = conv_b_index200;
    int conv_f_index010 = conv_b_index010;
    int conv_f_index011 = conv_b_index011;
    int conv_f_index210 = conv_b_index210;
    int conv_f_index201 = conv_b_index201;
    int conv_f_index211 = conv_b_index211;

    int conv_r_index000 = conv_f_index000;
    int conv_r_index001 = conv_f_index001;
    int conv_r_index010 = conv_f_index010;
    int conv_r_index011 = conv_f_index011;
    int conv_r_index100 = conv_f_index100;
    int conv_r_index101 = conv_f_index101;
    int conv_r_index110 = conv_f_index110;
    int conv_r_index111 = conv_f_index111;
    int conv_r_index200 = conv_f_index200;
    int conv_r_index201 = conv_f_index201;
    int conv_r_index210 = conv_f_index210;
    int conv_r_index211 = conv_f_index211;

		m_alpha_f0 	= _mm256_set1_pd(alphas[0]);
		m_alpha_f1 	= _mm256_set1_pd(alphas[1]);
		m_alpha_f2 	= _mm256_set1_pd(alphas[2]);

		m_bias_f0 	= _mm256_set1_pd(fil_b.data[0]);
		m_bias_f1 	= _mm256_set1_pd(fil_b.data[1]);
		m_bias_f2 	= _mm256_set1_pd(fil_b.data[2]);

		for (int r = 0, pool_i = 0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
		{
	    int conv_c_index000 = conv_r_index000;
	    int conv_c_index001 = conv_r_index001;
	    int conv_c_index010 = conv_r_index010;
	    int conv_c_index011 = conv_r_index011;
	    int conv_c_index100 = conv_r_index100;
	    int conv_c_index101 = conv_r_index101;
	    int conv_c_index110 = conv_r_index110;
	    int conv_c_index111 = conv_r_index111;
	    int conv_c_index200 = conv_r_index200;
	    int conv_c_index201 = conv_r_index201;
	    int conv_c_index210 = conv_r_index210;
	    int conv_c_index211 = conv_r_index211;

			for (int c = 0, pool_j = 0; c+1 < N_COLS_CONV; c=c+2, ++pool_j)
			{
				m_beta = _mm256_set_pd(betas[b][r+1][c+1], betas[b][r+1][c],
					 											betas[b][r][c+1], betas[b][r][c]);

				m_conv_val_f0 = _mm256_set1_pd(0.0);
				m_conv_val_f1 = _mm256_set1_pd(0.0);
				m_conv_val_f2 = _mm256_set1_pd(0.0);

        m_conv_val_ps_f0 = _mm256_set1_ps(0.0);
        m_conv_val_ps_f1 = _mm256_set1_ps(0.0);
        m_conv_val_ps_f2 = _mm256_set1_ps(0.0);

				for (int i = 0; i < FIL_ROWS; ++i)
				{

					for (int j = 0; j < FIL_COLS-1; j+=2)
					{
						// load filters
						j00 = fil_bin_w[0][i][j];
            			j10 = fil_bin_w[0][i][j+1];
            			j01 = fil_bin_w[1][i][j];
            			j11 = fil_bin_w[1][i][j+1];
            			j02 = fil_bin_w[2][i][j];
            			j12 = fil_bin_w[2][i][j+1];
						m_weight_ps_f0 = _mm256_set_ps(j10,j10,j10,j10,j00,j00,j00,j00);
						m_weight_ps_f1 = _mm256_set_ps(j11,j11,j11,j11,j01,j01,j01,j01);
						m_weight_ps_f2 = _mm256_set_ps(j12,j12,j12,j12,j02,j02,j02,j02);

						// load pixels from inputs
						m_input_pixel_ps = _mm256_set_ps(
              			bin_input_images[b][i+r+1][j+c+2],
              			bin_input_images[b][i+r+1][j+c+1],
              			bin_input_images[b][i+r][j+c+2],
              			bin_input_images[b][i+r][j+c+1],
              			bin_input_images[b][i+r+1][j+c+1],
              			bin_input_images[b][i+r+1][j+c],
              			bin_input_images[b][i+r][j+c+1],
              			bin_input_images[b][i+r][j+c]);

			            if(j00 == 1 && j10 == 1){
			              m_conv_val_ps_f0 += m_input_pixel_ps;
			            }
			            else if(j00 == -1 && j10 == -1){
			              m_conv_val_ps_f0 -= m_input_pixel_ps;
			            }
			            else if(j00 == 1){
			              m_conv_val_ps_f0 += _mm256_set_ps(
															-bin_input_images[b][i+r+1][j+c+2],
															-bin_input_images[b][i+r+1][j+c+1],
															-bin_input_images[b][i+r][j+c+2],
															-bin_input_images[b][i+r][j+c+1],
															bin_input_images[b][i+r+1][j+c+1],
															bin_input_images[b][i+r+1][j+c],
															bin_input_images[b][i+r][j+c+1],
															bin_input_images[b][i+r][j+c]
															);
			            }
			            else {
			              m_conv_val_ps_f0 += _mm256_set_ps(
              												bin_input_images[b][i+r+1][j+c+2],
              												bin_input_images[b][i+r+1][j+c+1],
              												bin_input_images[b][i+r][j+c+2],
              												bin_input_images[b][i+r][j+c+1],
              												-bin_input_images[b][i+r+1][j+c+1],
              												-bin_input_images[b][i+r+1][j+c],
              												-bin_input_images[b][i+r][j+c+1],
              												-bin_input_images[b][i+r][j+c]
              												);
			            }
			            
			            if(j01 == 1 && j11 == 1){
			              m_conv_val_ps_f1 += m_input_pixel_ps;
			            }
			            else if(j01 == -1 && j11 == -1){
			              m_conv_val_ps_f1 -= m_input_pixel_ps;
			            }
			            else if(j01 == 1){
			              m_conv_val_ps_f1 += _mm256_set_ps(
															-bin_input_images[b][i+r+1][j+c+2],
															-bin_input_images[b][i+r+1][j+c+1],
															-bin_input_images[b][i+r][j+c+2],
															-bin_input_images[b][i+r][j+c+1],
															bin_input_images[b][i+r+1][j+c+1],
															bin_input_images[b][i+r+1][j+c],
															bin_input_images[b][i+r][j+c+1],
															bin_input_images[b][i+r][j+c]
															);
			            }
			            else {
			              m_conv_val_ps_f1 += _mm256_set_ps(
              												bin_input_images[b][i+r+1][j+c+2],
              												bin_input_images[b][i+r+1][j+c+1],
              												bin_input_images[b][i+r][j+c+2],
              												bin_input_images[b][i+r][j+c+1],
              												-bin_input_images[b][i+r+1][j+c+1],
              												-bin_input_images[b][i+r+1][j+c],
              												-bin_input_images[b][i+r][j+c+1],
              												-bin_input_images[b][i+r][j+c]
              												);
			            }
			            
			            if(j02 == 1 && j12 == 1){
			              m_conv_val_ps_f2 += m_input_pixel_ps;
			            }
			            else if(j02 == -1 && j12 == -1){
			              m_conv_val_ps_f2 -= m_input_pixel_ps;
			            }
			            else if(j02 == 1){
			              m_conv_val_ps_f2 += _mm256_set_ps(
															-bin_input_images[b][i+r+1][j+c+2],
															-bin_input_images[b][i+r+1][j+c+1],
															-bin_input_images[b][i+r][j+c+2],
															-bin_input_images[b][i+r][j+c+1],
															bin_input_images[b][i+r+1][j+c+1],
															bin_input_images[b][i+r+1][j+c],
															bin_input_images[b][i+r][j+c+1],
															bin_input_images[b][i+r][j+c]
															);
			            }
			            else {
			              m_conv_val_ps_f2 += _mm256_set_ps(
              												bin_input_images[b][i+r+1][j+c+2],
              												bin_input_images[b][i+r+1][j+c+1],
              												bin_input_images[b][i+r][j+c+2],
              												bin_input_images[b][i+r][j+c+1],
              												-bin_input_images[b][i+r+1][j+c+1],
              												-bin_input_images[b][i+r+1][j+c],
              												-bin_input_images[b][i+r][j+c+1],
              												-bin_input_images[b][i+r][j+c]
              												);
			            }
						/*//do element wise product
						m_conv_val_ps_f0 = _mm256_add_ps(m_conv_val_ps_f0, _mm256_mul_ps(m_input_pixel_ps, m_weight_ps_f0));

						m_conv_val_ps_f1 = _mm256_add_ps(m_conv_val_ps_f1, _mm256_mul_ps(m_input_pixel_ps, m_weight_ps_f1));

						m_conv_val_ps_f2 = _mm256_add_ps(m_conv_val_ps_f2, _mm256_mul_ps(m_input_pixel_ps, m_weight_ps_f2));

						//store loaded pixels for reuse in next iteration*/
					}
          // load filters
          m_weight_f0 = _mm256_set1_pd(fil_bin_w[0][i][4]);
          m_weight_f1 = _mm256_set1_pd(fil_bin_w[1][i][4]);
          m_weight_f2 = _mm256_set1_pd(fil_bin_w[2][i][4]);

          // load pixels from inputs
          m_input_pixel = _mm256_set_pd(
            bin_input_images[b][i+r+1][c+5],
            bin_input_images[b][i+r+1][c+4],
            bin_input_images[b][i+r][c+5],
            bin_input_images[b][i+r][c+4]);

          // do element wise product
          m_conv_val_f0 = _mm256_add_pd(m_conv_val_f0, _mm256_mul_pd(m_input_pixel, m_weight_f0));
          m_conv_val_f1 = _mm256_add_pd(m_conv_val_f1, _mm256_mul_pd(m_input_pixel, m_weight_f1));
          m_conv_val_f2 = _mm256_add_pd(m_conv_val_f2, _mm256_mul_pd(m_input_pixel, m_weight_f2));
				}

        m_conv_val_f0 = _mm256_add_pd(m_conv_val_f0,
          _mm256_set_pd(m_conv_val_ps_f0[7]+m_conv_val_ps_f0[3],
          m_conv_val_ps_f0[6]+m_conv_val_ps_f0[2],
          m_conv_val_ps_f0[5]+m_conv_val_ps_f0[1],
          m_conv_val_ps_f0[4]+m_conv_val_ps_f0[0]));

        m_conv_val_f1 = _mm256_add_pd(m_conv_val_f1,
        _mm256_set_pd(m_conv_val_ps_f1[7]+m_conv_val_ps_f1[3],
          m_conv_val_ps_f1[6]+m_conv_val_ps_f1[2],
          m_conv_val_ps_f1[5]+m_conv_val_ps_f1[1],
          m_conv_val_ps_f1[4]+m_conv_val_ps_f1[0]));

        m_conv_val_f2 = _mm256_add_pd(m_conv_val_f2,
        _mm256_set_pd(m_conv_val_ps_f2[7]+m_conv_val_ps_f2[3],
          m_conv_val_ps_f2[6]+m_conv_val_ps_f2[2],
          m_conv_val_ps_f2[5]+m_conv_val_ps_f2[1],
          m_conv_val_ps_f2[4]+m_conv_val_ps_f2[0]));
        //*/

				// -----------------Filter 0------------------
				m_conv_val_f0 = _mm256_mul_pd(m_conv_val_f0, _mm256_mul_pd(m_alpha_f0, m_beta));
				m_conv_val_f0 = _mm256_add_pd(m_conv_val_f0, m_bias_f0);

				// -----------------Filter 1------------------
				m_conv_val_f1 = _mm256_mul_pd(m_conv_val_f1, _mm256_mul_pd(m_alpha_f1, m_beta));
				m_conv_val_f1 = _mm256_add_pd(m_conv_val_f1, m_bias_f1);
				// -----------------Filter 2------------------
				m_conv_val_f2 = _mm256_mul_pd(m_conv_val_f2, _mm256_mul_pd(m_alpha_f2, m_beta));
				m_conv_val_f2 = _mm256_add_pd(m_conv_val_f2, m_bias_f2);

				// ------------------------------------applying ReLU------------------------------------------
				// -----------------Filter 0------------------
				m_conv_val_f0 = _mm256_and_pd(m_conv_val_f0, _mm256_cmp_pd(m_conv_val_f0, zeroes_p, 0x0d));
				// -----------------Filter 1------------------
				m_conv_val_f1 = _mm256_and_pd(m_conv_val_f1, _mm256_cmp_pd(m_conv_val_f1, zeroes_p, 0x0d));
				// -----------------Filter 2------------------
				m_conv_val_f2 = _mm256_and_pd(m_conv_val_f2, _mm256_cmp_pd(m_conv_val_f2, zeroes_p, 0x0d));

				conv_val0_f0 = m_conv_val_f0[0];
				conv_val1_f0 = m_conv_val_f0[1];
				conv_val2_f0 = m_conv_val_f0[2];
				conv_val3_f0 = m_conv_val_f0[3];
				conv_val0_f1 = m_conv_val_f1[0];
				conv_val1_f1 = m_conv_val_f1[1];
				conv_val2_f1 = m_conv_val_f1[2];
				conv_val3_f1 = m_conv_val_f1[3];
				conv_val0_f2 = m_conv_val_f2[0];
				conv_val1_f2 = m_conv_val_f2[1];
				conv_val2_f2 = m_conv_val_f2[2];
				conv_val3_f2 = m_conv_val_f2[3];

				conv_data[conv_c_index000] = conv_val0_f0;
				conv_data[conv_c_index001] = conv_val1_f0;
				conv_data[conv_c_index010] = conv_val2_f0;
				conv_data[conv_c_index011] = conv_val3_f0;
				conv_data[conv_c_index100] = conv_val0_f1;
				conv_data[conv_c_index101] = conv_val1_f1;
				conv_data[conv_c_index110] = conv_val2_f1;
				conv_data[conv_c_index111] = conv_val3_f1;
				conv_data[conv_c_index200] = conv_val0_f2;
				conv_data[conv_c_index201] = conv_val1_f2;
				conv_data[conv_c_index210] = conv_val2_f2;
				conv_data[conv_c_index211] = conv_val3_f2;

				// --------Max pooling---------------

				INCREMENT_FLOPS(9)

				// -----------------Filter 0------------------
				if ( conv_val0_f0 > conv_val1_f0)
				{
					max_pool_1_f0 = conv_val0_f0;
					ind_1_i_f0 = r;
					ind_1_j_f0 = c;
				}
				else
				{
					max_pool_1_f0 = conv_val1_f0;
					ind_1_i_f0 = r  ;
					ind_1_j_f0 = c+1;
				}

				if (conv_val2_f0 > conv_val3_f0)
				{
					max_pool_2_f0 = conv_val2_f0;
					ind_2_i_f0 = r+1;
					ind_2_j_f0 = c  ;
				}
				else
				{
					max_pool_2_f0 = conv_val3_f0;
					ind_2_i_f0 = r+1;
					ind_2_j_f0 = c+1;
				}

				if (max_pool_1_f0 > max_pool_2_f0)
				{
					max_pool_f0 = max_pool_1_f0;
					ind_i_f0 = ind_1_i_f0;
					ind_j_f0 = ind_1_j_f0;
				}
				else
				{
					max_pool_f0 = max_pool_2_f0;
					ind_i_f0 = ind_2_i_f0;
					ind_j_f0 = ind_2_j_f0;
				}

				// -----------------Filter 1------------------
				if (conv_val0_f1 > conv_val1_f1)
				{
					max_pool_1_f1 = conv_val0_f1;
					ind_1_i_f1 = r;
					ind_1_j_f1 = c;
				}
				else
				{
					max_pool_1_f1 = conv_val1_f1;
					ind_1_i_f1 = r  ;
					ind_1_j_f1 = c+1;
				}

				if (conv_val2_f1 > conv_val3_f1)
				{
					max_pool_2_f1 = conv_val2_f1;
					ind_2_i_f1 = r+1;
					ind_2_j_f1 = c  ;
				}
				else
				{
					max_pool_2_f1 = conv_val3_f1;
					ind_2_i_f1 = r+1;
					ind_2_j_f1 = c+1;
				}

				if (max_pool_1_f1 > max_pool_2_f1)
				{
					max_pool_f1 = max_pool_1_f1;
					ind_i_f1 = ind_1_i_f1;
					ind_j_f1 = ind_1_j_f1;
				}
				else
				{
					max_pool_f1 = max_pool_2_f1;
					ind_i_f1 = ind_2_i_f1;
					ind_j_f1 = ind_2_j_f1;
				}

				// -----------------Filter 2------------------
				if (conv_val0_f2 > conv_val1_f2)
				{
					max_pool_1_f2 = conv_val0_f2;
					ind_1_i_f2 = r;
					ind_1_j_f2 = c;
				}
				else
				{
					max_pool_1_f2 = conv_val1_f2;
					ind_1_i_f2 = r  ;
					ind_1_j_f2 = c+1;
				}

				if (conv_val2_f2 > conv_val3_f2)
				{
					max_pool_2_f2 = conv_val2_f2;
					ind_2_i_f2 = r+1;
					ind_2_j_f2 = c  ;
				}
				else
				{
					max_pool_2_f2 = conv_val3_f2;
					ind_2_i_f2 = r+1;
					ind_2_j_f2 = c+1;
				}

				if (max_pool_1_f2 > max_pool_2_f2)
				{
					max_pool_f2 = max_pool_1_f2;
					ind_i_f2 = ind_1_i_f2;
					ind_j_f2 = ind_1_j_f2;
				}
				else
				{
					max_pool_f2 = max_pool_2_f2;
				  ind_i_f2 = ind_2_i_f2;
				  ind_j_f2 = ind_2_j_f2;
				}

				pool_data[ind_pool_out(b, 0, pool_i, pool_j)] = max_pool_f0;
				pool_data[ind_pool_out(b, 1, pool_i, pool_j)] = max_pool_f1;
				pool_data[ind_pool_out(b, 2, pool_i, pool_j)] = max_pool_f2;

				pool_index_i[b][0][pool_i][pool_j] = ind_i_f0;
				pool_index_j[b][0][pool_i][pool_j] = ind_j_f0;

				pool_index_i[b][1][pool_i][pool_j] = ind_i_f1;
				pool_index_j[b][1][pool_i][pool_j] = ind_j_f1;

				pool_index_i[b][2][pool_i][pool_j] = ind_i_f2;
				pool_index_j[b][2][pool_i][pool_j] = ind_j_f2;

	      conv_c_index000 += 2;
	      conv_c_index001 += 2;
	      conv_c_index010 += 2;
	      conv_c_index011 += 2;
	      conv_c_index100 += 2;
	      conv_c_index101 += 2;
	      conv_c_index110 += 2;
	      conv_c_index111 += 2;
	      conv_c_index200 += 2;
	      conv_c_index201 += 2;
	      conv_c_index210 += 2;
	      conv_c_index211 += 2;
				}
	    conv_r_index000 += _2_N_COLS_CONV;
	    conv_r_index001 += _2_N_COLS_CONV;
	    conv_r_index010 += _2_N_COLS_CONV;
	    conv_r_index011 += _2_N_COLS_CONV;
	    conv_r_index100 += _2_N_COLS_CONV;
	    conv_r_index101 += _2_N_COLS_CONV;
	    conv_r_index110 += _2_N_COLS_CONV;
	    conv_r_index111 += _2_N_COLS_CONV;
	    conv_r_index200 += _2_N_COLS_CONV;
	    conv_r_index201 += _2_N_COLS_CONV;
	    conv_r_index210 += _2_N_COLS_CONV;
	    conv_r_index211 += _2_N_COLS_CONV;
			}
	  conv_f_index100 += THREE_FIL_CONV_SIZE;
	  conv_f_index000 += THREE_FIL_CONV_SIZE;
	  conv_f_index001 += THREE_FIL_CONV_SIZE;
	  conv_f_index110 += THREE_FIL_CONV_SIZE;
	  conv_f_index101 += THREE_FIL_CONV_SIZE;
	  conv_f_index111 += THREE_FIL_CONV_SIZE;
	  conv_f_index200 += THREE_FIL_CONV_SIZE;
	  conv_f_index010 += THREE_FIL_CONV_SIZE;
	  conv_f_index011 += THREE_FIL_CONV_SIZE;
	  conv_f_index210 += THREE_FIL_CONV_SIZE;
	  conv_f_index201 += THREE_FIL_CONV_SIZE;
	  conv_f_index211 += THREE_FIL_CONV_SIZE;

  conv_b_index000 += CONV_SIZE;
  conv_b_index001 += CONV_SIZE;
  conv_b_index010 += CONV_SIZE;
  conv_b_index011 += CONV_SIZE;
  conv_b_index100 += CONV_SIZE;
  conv_b_index101 += CONV_SIZE;
  conv_b_index110 += CONV_SIZE;
  conv_b_index111 += CONV_SIZE;
  conv_b_index200 += CONV_SIZE;
  conv_b_index201 += CONV_SIZE;
  conv_b_index210 += CONV_SIZE;
  conv_b_index211 += CONV_SIZE;
	}
}

// loop on conv rows and cols unrolled by 2, max-pooling done, pool indexes not saved
void xnor_convolve_pool_validation(int bin_input_images[BATCH_SIZE][IMAGE_ROWS][IMAGE_COLS], double betas[BATCH_SIZE][N_ROWS_CONV][N_COLS_CONV],
						tensor* conv_t, int batch_size, int fil_bin_w[NUM_FILS][FIL_ROWS][FIL_COLS],
						double alphas[NUM_FILS], tensor fil_b, tensor* pool_t)
{
	double conv_val0, conv_val1, conv_val2, conv_val3;
	double alpha, bias;
	double beta0, beta1, beta2, beta3;
	double weight;

	double prev1, prev2, curr1, curr2;

	double input_pixel0;
	double input_pixel1;
	double input_pixel2;
	double input_pixel3;

	int pool_i, pool_j;
	int max_pool_1, max_pool_2, max_pool;

	for (int b = 0; b < batch_size; ++b)
	{

		for (int f = 0; f < NUM_FILS; ++f)
		{

			alpha = alphas[f];
			bias = fil_b.data[f];

			for (int r = 0, pool_i = 0; r+1 < N_ROWS_CONV; r=r+2, ++pool_i)
			{

				for (int c = 0, pool_j = 0; c+1 < N_COLS_CONV; c=c+2, ++pool_j)
				{

					beta0 = betas[b][r  ][c  ];
					beta1 = betas[b][r  ][c+1];
					beta2 = betas[b][r+1][c  ];
					beta3 = betas[b][r+1][c+1];

					conv_val0 = 0.0;
					conv_val1 = 0.0;
					conv_val2 = 0.0;
					conv_val3 = 0.0;

					for (int i = 0; i < FIL_ROWS; ++i)
					{

						prev1 = bin_input_images[b][i+r  ][0+c  ];
						prev2 = bin_input_images[b][i+r+1][0+c  ];

						for (int j = 0; j < FIL_COLS; ++j)
						{

							INCREMENT_FLOPS(8)

							weight = fil_bin_w[f][i][j];

							input_pixel0 = prev1;
							input_pixel1 = bin_input_images[b][i+r  ][j+c+1];
							input_pixel2 = prev2;
							input_pixel3 = bin_input_images[b][i+r+1][j+c+1];

							// XNOR operation
							//conv_val += ( bin_input_images[b][i+r][j+c] == fil_bin_w[f][i][j] );

							conv_val0 += ( input_pixel0 * weight );
							conv_val1 += ( input_pixel1 * weight );
							conv_val2 += ( input_pixel2 * weight );
							conv_val3 += ( input_pixel3 * weight );

							prev1 = input_pixel1;
							prev2 = input_pixel3;
						}
					}

					INCREMENT_FLOPS(16)

					conv_val0 *= alpha * beta0;
					conv_val1 *= alpha * beta1;
					conv_val2 *= alpha * beta2;
					conv_val3 *= alpha * beta3;

					conv_val0 += bias;
					conv_val1 += bias;
					conv_val2 += bias;
					conv_val3 += bias;

					// applying ReLU
					if (conv_val0 < 0.0)
					{
						conv_val0 = 0.0;
					}

					if (conv_val1 < 0.0)
					{
						conv_val1 = 0.0;
					}

					if (conv_val2 < 0.0)
					{
						conv_val2 = 0.0;
					}

					if (conv_val3 < 0.0)
					{
						conv_val3 = 0.0;
					}

					(conv_t->data)[ind_conv_out(b, f, r  , c  )] = conv_val0;
					(conv_t->data)[ind_conv_out(b, f, r  , c+1)] = conv_val1;
					(conv_t->data)[ind_conv_out(b, f, r+1, c  )] = conv_val2;
					(conv_t->data)[ind_conv_out(b, f, r+1, c+1)] = conv_val3;


					// ----------------------------------------------Max pooling---------------------------------------

					if (conv_val0 > conv_val1)
					{
						max_pool_1 = conv_val0;
					}
					else
					{
						max_pool_1 = conv_val1;
					}

					if (conv_val2 > conv_val3)
					{
						max_pool_2 = conv_val2;
					}
					else
					{
						max_pool_2 = conv_val3;
					}

					if (max_pool_1 > max_pool_2)
					{
						max_pool = max_pool_1;
					}
					else
					{
						max_pool = max_pool_2;
					}

					(pool_t->data)[ind_pool_out(b, f, pool_i, pool_j)] = max_pool;
				}
			}

		}
	}
}


void initialize_filters(tensor* fil_w, tensor* fil_b)
{
    srand(time(NULL));

    for(int k = 0; k < NUM_FILS; k++){
        (fil_b->data)[offset(fil_b, k, 0, 0, 0)] = 0.0;
        for(int i = 0; i < FIL_ROWS; ++i){
            for(int j = 0; j < FIL_COLS; ++j){
                for(int l = 0; l < FIL_DEPTH; ++l){
                    int r = rand();
                    double ran = ((double)rand())/RAND_MAX;
                    if (r%2 == 0){
                        (fil_w->data)[offset(fil_w, k, j, i, l)] = -ran;
                    }else{
                        (fil_w->data)[offset(fil_w, k, j, i, l)] = ran;
                    }
                }
            }
        }
    }
}

void print_filters(tensor* fil_w, tensor* fil_b)
{
	for (int k = 0; k < NUM_FILS; ++k)
	{
		printf("k=%d, bias=%f\nweights:\n", k, (fil_b->data)[offset(fil_b, k, 0, 0, 0)]);
		for (int i = 0; i < FIL_ROWS; ++i)
		{
			for (int j = 0; j < FIL_COLS; ++j)
			{
				printf("%.3f, ", (fil_w->data)[offset(fil_w, k, j, i, 0)]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void print_bin_filters(int bin_fil_w[NUM_FILS][FIL_ROWS][FIL_COLS], double alphas[NUM_FILS])
{
	for (int k = 0; k < NUM_FILS; ++k)
	{
		printf("k=%d, alpha=%f\nweights:\n", k, alphas[k]);
		for (int i = 0; i < FIL_ROWS; ++i)
		{
			for (int j = 0; j < FIL_COLS; ++j)
			{
				printf("%3d, ", bin_fil_w[k][i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}
