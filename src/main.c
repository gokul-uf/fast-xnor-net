#include <stdio.h>
#include "main.h"
#include "xnornet.h"
#include "tensor.h"

int main(){
    printf("starting program\n");
    tensor t;
    build(&t);
    t.data[offset(&t,0,0,0,0)] = 1;
    t.data[offset(&t,0,0,1,0)] = 4;
    t.data[offset(&t,0,1,0,0)] = 3;
    t.data[offset(&t,0,1,1,0)] = 4;
    
    printf("%d %d %d %d ---> %d x %d x %d x %d \n", t.data[0], t.data[1], t.data[2], t.data[3],
           t.width, t.height, t.depth, t.batch_size);
    destroy(&t);
    return 0;
}
