#ifndef COST_H
#define COST_H

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

#define COST_T    uint64_t
#define PRI_COST PRIu64

#if COST_MEASURE

extern COST_T __costF_load;
extern COST_T __costF_store;
extern COST_T __costF_add;
extern COST_T __costF_mul;
extern COST_T __costF_div;
extern COST_T __costF_max;
extern COST_T __costF_other;

extern COST_T __costI_load;
extern COST_T __costI_store;
extern COST_T __costI_add;
extern COST_T __costI_mul;
extern COST_T __costI_div;
extern COST_T __costI_max;
extern COST_T __costI_other;

#define COST_VARIABLES_HERE   \
    COST_T __costF_load = 0;  \
    COST_T __costF_store = 0; \
    COST_T __costF_add = 0;   \
    COST_T __costF_mul = 0;   \
    COST_T __costF_div = 0;   \
    COST_T __costF_max = 0;   \
    COST_T __costF_other = 0; \
    COST_T __costI_load = 0;  \
    COST_T __costI_store = 0; \
    COST_T __costI_add = 0;   \
    COST_T __costI_mul = 0;   \
    COST_T __costI_div = 0;   \
    COST_T __costI_max = 0;   \
    COST_T __costI_other = 0;

#define COST_RESET            \
    __costF_load = 0;  \
    __costF_store = 0; \
    __costF_add = 0;   \
    __costF_mul = 0;   \
    __costF_div = 0;   \
    __costF_max = 0;   \
    __costF_other = 0; \
    __costI_load = 0;  \
    __costI_store = 0; \
    __costI_add = 0;   \
    __costI_mul = 0;   \
    __costI_div = 0;   \
    __costI_max = 0;   \
    __costI_other = 0;

#define COST_F_LOAD ((const COST_T) __costF_load)
#define COST_F_STORE ((const COST_T) __costF_store)
#define COST_F_ADD ((const COST_T) __costF_add)
#define COST_F_MUL ((const COST_T) __costF_mul)
#define COST_F_DIV ((const COST_T) __costF_div)
#define COST_F_MAX ((const COST_T) __costF_max)
#define COST_F_OTHER ((const COST_T) __costF_other)

#define COST_I_LOAD ((const COST_T) __costI_load)
#define COST_I_STORE ((const COST_T) __costI_store)
#define COST_I_ADD ((const COST_T) __costI_add)
#define COST_I_MUL ((const COST_T) __costI_mul)
#define COST_I_DIV ((const COST_T) __costI_div)
#define COST_I_MAX ((const COST_T) __costI_max)
#define COST_I_OTHER ((const COST_T) __costI_other)


#define COST_INC_F_LOAD(x) {__costF_load += (x);}
#define COST_INC_F_STORE(x) {__costF_store += (x);}
#define COST_INC_F_ADD(x) {__costF_add += (x);}
#define COST_INC_F_MUL(x) {__costF_mul += (x);}
#define COST_INC_F_DIV(x) {__costF_div += (x);}
#define COST_INC_F_MAX(x) {__costF_max += (x);}
#define COST_INC_F_OTHER(x) {__costF_other += (x);}

#define COST_INC_I_LOAD(x) {__costI_load += (x);}
#define COST_INC_I_STORE(x) {__costI_store += (x);}
#define COST_INC_I_ADD(x) {__costI_add += (x);}
#define COST_INC_I_MUL(x) {__costI_mul += (x);}
#define COST_INC_I_DIV(x) {__costI_div += (x);}
#define COST_INC_I_MAX(x) {__costI_max += (x);}
#define COST_INC_I_OTHER(x) {__costI_other += (x);}

#else

// add all things empty here

#define COST_VARIABLES_HERE
#define COST_MODEL_RESET

#define COST_F_LOAD ((const COST_T) 0)
#define COST_F_STORE ((const COST_T) 0)
#define COST_F_ADD ((const COST_T) 0)
#define COST_F_MUL ((const COST_T) 0)
#define COST_F_DIV ((const COST_T) 0)
#define COST_F_MAX ((const COST_T) 0)
#define COST_F_OTHER ((const COST_T) 0)

#define COST_I_LOAD ((const COST_T) 0)
#define COST_I_STORE ((const COST_T) 0)
#define COST_I_ADD ((const COST_T) 0)
#define COST_I_MUL ((const COST_T) 0)
#define COST_I_DIV ((const COST_T) 0)
#define COST_I_MAX ((const COST_T) 0)
#define COST_I_OTHER ((const COST_T) 0)

#define COST_INC_F_LOAD(x)
#define COST_INC_F_STORE(x)
#define COST_INC_F_ADD(x)
#define COST_INC_F_MUL(x)
#define COST_INC_F_DIV(x)
#define COST_INC_F_MAX(x)
#define COST_INC_F_OTHER(x)

#define COST_INC_I_LOAD(x)
#define COST_INC_I_STORE(x)
#define COST_INC_I_ADD(x)
#define COST_INC_I_MUL(x)
#define COST_INC_I_DIV(x)
#define COST_INC_I_MAX(x)
#define COST_INC_I_OTHER(x)

#endif
#endif // COST_H
