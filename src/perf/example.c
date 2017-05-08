#include <stdlib.h>
#include <stdio.h>

// important - for this to work, these 2 need to be here:
#include "common_cost.h"

COST_VARIABLES_HERE

extern void aux_code();
void print_cost_model(void);

int main(){
  double a_f = 1.0;
  double b_f = 2.0;
  double c_f = 3.0;

  int a_i = 1.0;
  int b_i = 2.0;
  int c_i = 3.0;

  int i, j;

  for(i=0; i<20; ++i){
    for(j=0; j<20; ++j){
      COST_INC_F_ADD(1); COST_INC_F_MUL(1);
      a_f = a_f * b_f + c_f;
    }
  }

  for(i=0; i<10; ++i){
    for(j=0; j<10; ++j){
      COST_INC_I_ADD(1); COST_INC_I_MUL(1);
      a_i = a_i * b_i + c_i;
    }
  }


  printf("\nRunning aux_code...\n");
  aux_code();

  print_cost_model();

  COST_RESET

  aux_code();

  print_cost_model();
}


void print_cost_model(){
  printf("Cost Model FLOATS:\n");
  printf("  Add: %"PRI_COST"\n", COST_F_ADD);
  printf("  Mul: %"PRI_COST"\n", COST_F_MUL);
  printf("  Div: %"PRI_COST"\n", COST_F_DIV);
  printf("  Max: %"PRI_COST"\n", COST_F_MAX);
  printf("  Oth: %"PRI_COST"\n", COST_F_OTHER);

  printf("\nCost Model INTS:\n");
  printf("  Add: %"PRI_COST"\n", COST_I_ADD);
  printf("  Mul: %"PRI_COST"\n", COST_I_MUL);
  printf("  Div: %"PRI_COST"\n", COST_I_DIV);
  printf("  Max: %"PRI_COST"\n", COST_I_MAX);
  printf("  Oth: %"PRI_COST"\n", COST_I_OTHER);
}
