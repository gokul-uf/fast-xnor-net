#ifdef WIN32
    #include <windows.h>
#endif
#ifdef __linux__
    #include <unistd.h>
#endif
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <string>
#include "../../include/perf.h"
#include "../pcm/cpucounters.h"
#include "asm.h"

using namespace std;

#ifdef __APPLE__
    typedef union {
        uint64_t v64;
        struct {
            uint32_t lo;
            uint32_t hi;
        } v32;
    } mac_tsc_counter_t;

    mac_tsc_counter_t mac_tsc_counter_start;
    mac_tsc_counter_t mac_tsc_counter_stop;
#else
    uint64 cpu_clk_unhalted_thread_counter_start = 0;
    uint64 cpu_clk_unhalted_thread_counter_stop = 0;
    MsrHandle * cpu_core_msr_handle;
#endif

PCM * intel_PCM_instance;

void start_intel_pcm () {
    cout << "==============================================================" << endl;
    cout << "= IntelPCM is about to be initialized" << endl;
    cout << "==============================================================" << endl;
    intel_PCM_instance = PCM::getInstance();
    intel_PCM_instance->resetPMU();
    PCM::ErrorCode status = intel_PCM_instance->program();
    if (status == PCM::Success) {
        cout << "==============================================================" << endl;
        cout << endl;
        cout << "==============================================================" << endl;
        cout << "= IntelPCM initialized" << endl;
        cout << "==============================================================" << endl;
        cout << "Detected: " << intel_PCM_instance->getCPUBrandString() << endl;
        cout << "Codename: " << intel_PCM_instance->getUArchCodename() << endl;
        cout << "Stepping: " << intel_PCM_instance->getCPUStepping() << endl;
        cout << "==============================================================" << endl;
        cout << endl << endl;
    } else {
        cout << "Access to Intel(r) Performance Counter Monitor has been denied" << endl;
        exit(EXIT_FAILURE);
    }
}


void perf_init ()
{
    start_intel_pcm ();

    int num_cores = intel_PCM_instance->getNumCores();
    int core;
    for (core = num_cores - 1; core >= 0; core -= 1) {
        if ( intel_PCM_instance->isCoreOnline(core) ) {
            break;
        }
    }

    #ifdef __linux__
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        CPU_SET(core, &cpu_set);
        sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpu_set);
        cpu_core_msr_handle = new MsrHandle(core);
    #endif

    #ifdef WIN32
        HANDLE process = GetCurrentProcess();
        DWORD_PTR processAffinityMask = (1 << core);
        BOOL success = SetProcessAffinityMask(process, processAffinityMask);
        cpu_core_msr_handle = new MsrHandle(core);
    #endif
}


void cycles_count_start ()
{
    #ifdef __APPLE__
        ASM VOLATILE ("cpuid" : : "a" (0) : "bx", "cx", "dx" );
        ASM VOLATILE ("rdtsc" : "=a" ((mac_tsc_counter_start).v32.lo), "=d"((mac_tsc_counter_start).v32.hi));
    #else
        cpu_core_msr_handle->read(CPU_CLK_UNHALTED_THREAD_ADDR, &cpu_clk_unhalted_thread_counter_start);
    #endif
}

uint64_t cycles_count_stop () {
    #ifdef __APPLE__
        ASM VOLATILE ("rdtsc" : "=a" ((mac_tsc_counter_stop).v32.lo), "=d"((mac_tsc_counter_stop).v32.hi));
        ASM VOLATILE ("cpuid" : : "a" (0) : "bx", "cx", "dx" );
        return mac_tsc_counter_stop.v64 - mac_tsc_counter_start.v64;
    #else
        cpu_core_msr_handle->read(CPU_CLK_UNHALTED_THREAD_ADDR, &cpu_clk_unhalted_thread_counter_stop);
        return (uint64_t)(cpu_clk_unhalted_thread_counter_stop - cpu_clk_unhalted_thread_counter_start);
    #endif

    return 0;
}

void perf_done ()
{
#if defined (WIN32)
    char line[256];
    printf("\n\nPress any key to exit. \n");
    fgets(line, sizeof line, stdin);
#endif
}



