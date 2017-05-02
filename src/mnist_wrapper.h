#ifndef MNIST_WRAPPER_H
#define MNIST_WRAPPER_H

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 

#define PRINT_INT_TO_BINARY(i) \
  printf("i: "BYTE_TO_BINARY_PATTERN" "BYTE_TO_BINARY_PATTERN" "BYTE_TO_BINARY_PATTERN" "BYTE_TO_BINARY_PATTERN"\n", \
	BYTE_TO_BINARY(i>>24), BYTE_TO_BINARY(i>>16), BYTE_TO_BINARY(i>>8), BYTE_TO_BINARY(i));

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "tensor.h"
#include "stddef.h"

extern char* TRAIN_IMAGES;
extern char* TRAIN_LABELS;
extern char* TEST_IMAGES;
extern char* TEST_LABELS;

int ReverseInt (int i);
void read_mnist_images_labels(char* images_path, char* labels_path , int* number_of_images, int* number_of_labels, 
	int* n_rows, int* n_cols, tensor* input_tensor, int** labels);
void test_mnist_load(tensor t, int* labels, int number);
void test_reverse_int();
void set_paths();

#endif