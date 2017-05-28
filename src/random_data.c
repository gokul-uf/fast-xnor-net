#include "random_data.h"

void random_data(int image_cols, int image_rows, tensor* input_tensor, int** labels){
  NUM_IMAGES = 60000;
  NUM_LABELS = NUM_IMAGES;
  IMAGE_COLS = image_cols;
  IMAGE_ROWS = image_rows;

  build_args(input_tensor, IMAGE_COLS, IMAGE_ROWS, 1, NUM_IMAGES);
  *labels = malloc(NUM_LABELS * sizeof(int));

  for (int b = 0; b < NUM_IMAGES; ++b)
  {
    for (int w = 0; w < IMAGE_COLS; ++w)
    {
      for (int h = 0; h < IMAGE_ROWS; ++h)
      {
        unsigned char temp_char = rand() % 256;
        (input_tensor->data)[offset(input_tensor,b,w,h,0)] = ((double)temp_char - 127)/127.0;
      }
    }
  }

  for (int i = 0; i < NUM_LABELS; ++i)
  {
    (*labels)[i] = (int)(rand() % 10);
  }

}
