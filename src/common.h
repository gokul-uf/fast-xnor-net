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
#define PRINT_FLOPS()  			printf("Total flops=2114265\n"); 
#define PRINT_PERF(cycles)      printf("performace =%f flops/cycle\n", 1.0*2114265/cycles);

#endif

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
