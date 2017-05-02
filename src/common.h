#ifndef COMMON_H
#define COMMON_H

#define BATCH_SIZE 100
#define NUM_FILS 3
#define FIL_ROWS 5
#define FIL_COLS 5
#define POOL_DIM 2
#define N_DIGS 10

#include <stdio.h>

extern char* TRAIN_IMAGES;
extern char* TRAIN_LABELS;
extern char* TEST_IMAGES;
extern char* TEST_LABELS;
extern int N_ROWS_CONV;
extern int N_COLS_CONV;
extern int N_ROWS_POOL;
extern int N_COLS_POOL;

#endif