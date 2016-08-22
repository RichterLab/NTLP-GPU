#include "stdio.h"

extern "C" int gpudevices(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf("Actual: %d\n", nDevices);
    return nDevices;
}
