extern "C" int gpudevices(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    return nDevices;
}
