#include <cstdint>
#include <iostream>
#include <string>
#include "../../include/perf.h"
#include "asm.h"

using namespace std;

#ifndef WIN32
    #define myInt64 unsigned long long
    #define INT32 unsigned int
#else
    #define myInt64 signed __int64
	#define INT32 unsigned __int32
#endif


#if defined(WIN32) || defined(_MSC_VER)
    typedef union
	{
	    myInt64 int64;
        struct {
            INT32 lo;
            INT32 hi;
        } int32;
	} tsc_counter_t;

	#define RDTSC(cpu_c)   \
	{       __asm rdtsc    \
			__asm mov (cpu_c).int32.lo,eax  \
			__asm mov (cpu_c).int32.hi,edx  \
	}

	#define CPUID() \
	{ \
		__asm mov eax, 0 \
		__asm cpuid \
	}
#else
    typedef union {
        myInt64 int64;
        struct {
            INT32 lo;
            INT32 hi;
        } int32;
    } tsc_counter_t;

    #define RDTSC(cpu_c) ASM VOLATILE ("rdtsc" : "=a" ((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))
    #define CPUID()      ASM VOLATILE ("cpuid" : : "a" (0) : "bx", "cx", "dx" )
#endif

tsc_counter_t failback_mode_tsc_counter_start;
tsc_counter_t failback_mode_tsc_counter_stop;

void perf_init ()
{
    cout << "==============================================================" << endl;
    cout << "= Running in Failback mode" << endl;
    cout << "==============================================================" << endl;
}


void cycles_count_start ()
{
    CPUID();
    RDTSC(failback_mode_tsc_counter_start);
}

uint64_t cycles_count_stop () {
    RDTSC(failback_mode_tsc_counter_stop);
    CPUID();
    return (uint64_t)(failback_mode_tsc_counter_stop.int64 - failback_mode_tsc_counter_start.int64);
}

void perf_done ()
{
    // do nuttin for now.
}
