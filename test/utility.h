#ifndef TEST_UTILITY_H_
#define TEST_UTILITY_H_

#include <string>
#include <vector>
#include "particle_gpu.h"

std::vector<Particle> ReadParticles( std::string path );
double* ReadArray(const char* path, unsigned int *size);

#endif // TEST_UTILITY_H_