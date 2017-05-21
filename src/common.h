#ifndef COMMON_H
#define COMMON_H

#define BATCH_SIZE 10
#define NUM_VAL 10000
#define NUM_FILS 3
#define FIL_ROWS 5
#define FIL_COLS 5
#define FIL_DEPTH 1
#define POOL_DIM 2
#define N_DIGS 10
#define LEARN_RATE 0.01
#define BINARY_NET 1
//#define COUNT_FLOPS

// conditional compilation of flop counts

// Binary net no unrolling flops: 2166176
//   XNOR net no unrolling flops: 3061056

#ifdef COUNT_FLOPS
#define NUM_EPOCHS 				1
#define COUNT_BATCHES			1
#define INCREMENT_FLOPS(i)		TOTAL_FLOPS += i;
#define PRINT_FLOPS()			printf("Total flops=%d\n", TOTAL_FLOPS);
#define PRINT_PERF(cycles)		printf("performace =%f flops/cycle\n", 1.0*TOTAL_FLOPS/cycles);

#else
#define NUM_EPOCHS 				20
#define COUNT_BATCHES			NUM_TRAIN/BATCH_SIZE
#define INCREMENT_FLOPS(i)		;
#define PRINT_FLOPS()  			printf("Total flops=1828154\n"); //binary net = 1828154, xnor net = 2442874
#define PRINT_PERF(cycles)      printf("performace =%f flops/cycle\n", 1.0*1828154/cycles);

#endif

// -----------------------------MACROS for indexing tensors------------------------------------------

#define ind_fil_w(f, r, c)	\
	( ( (f) * FIL_ROWS*FIL_COLS ) + ( (r) * FIL_COLS ) + (c) )

#define ind_input_img(b, r, c)	\
	( ( (b) * IMAGE_ROWS*IMAGE_COLS ) + ( (r) *IMAGE_COLS ) + (c) )

#define ind_conv_out(b, f, r, c)	\
	( ( (b) * NUM_FILS*N_ROWS_CONV*N_COLS_CONV ) + ( (f) * N_ROWS_CONV*N_COLS_CONV ) + ( (r) * N_COLS_CONV ) + (c) )

#define ind_pool_out(b, f, r, c)	\
	( ( (b) * NUM_FILS*N_ROWS_POOL*N_COLS_POOL ) + ( (f) * N_ROWS_POOL*N_COLS_POOL ) + ( (r) * N_COLS_POOL ) + (c) )

#define ind_fully_con_w(d, f, r, c)	\
	( ( (d) * NUM_FILS*N_ROWS_POOL*N_COLS_POOL ) + ( (f) * N_ROWS_POOL*N_COLS_POOL ) + ( (r) * N_COLS_POOL ) + (c) )

#define ind_fully_con_out(b, d)	\
	( ( (b) * N_DIGS ) + (d) )

#define ind_softmax_out(b, d)	\
	( ( (b) * N_DIGS ) + (d) )

// --------------------------------------------------------------------------------------------------

#include <stdio.h>

extern char* TRAIN_IMAGES;
extern char* TRAIN_LABELS;
extern char* TEST_IMAGES;
extern char* TEST_LABELS;
extern int NUM_IMAGES;
extern int NUM_LABELS;
extern int IMAGE_ROWS;
extern int IMAGE_COLS;
extern int N_ROWS_CONV;
extern int N_COLS_CONV;
extern int N_ROWS_POOL;
extern int N_COLS_POOL;
extern int NUM_TRAIN;
extern int N_BATCHES;
extern int TOTAL_FLOPS;
extern double MULTIPLIER;

#endif
