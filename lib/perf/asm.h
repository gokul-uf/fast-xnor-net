#ifndef ASM_H
#define ASM_H

#ifndef WIN32
    #if defined(__GNUC__) || defined(__linux__)
        #define VOLATILE __volatile__
        #define ASM __asm__
    #else
        #define ASM asm
		#define VOLATILE
    #endif
#endif


#endif /* ASM_H */