#include <stdlib.h>
#include <stdio.h>

// important - for this to work, these 2 need to be here:
#include "common_cost.h"

#define N 20

double a = 10;
double b = 9;
double c = 8;

int i;

void aux_code(){
  for(i=0; i<N; ++i){
    COST_INC_F_ADD(1); COST_INC_F_MUL(2); COST_INC_F_DIV(1);
    c = (3.0*a) / b + 2.0*c;
  }
}
