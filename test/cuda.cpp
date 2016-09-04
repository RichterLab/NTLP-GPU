#include "gtest/gtest.h"
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>

#include "particle_gpu.h"
#include "utility.h"

TEST( ParticleCUDA, Random ) {
    ASSERT_DOUBLE_EQ(rand2(1080), 0.65541634834855311);
    ASSERT_DOUBLE_EQ(rand2(1080), 0.20099518545185716);
}

TEST( ParticleCUDA, ParticleUpdate ) {
    std::vector<Particle> input = ReadParticles( "../test/data/particle_input.dat" );
	ASSERT_EQ( input.size(), 10 );

	std::vector<Particle> expected = ReadParticles( "../test/data/particle_expected.dat" );
	ASSERT_EQ( expected.size(), 10 );

	GPU *gpu = (GPU*) malloc( sizeof(GPU) );
	ParticleInit(gpu, 10, &input[0]);
	ParticleStep(gpu, 500, 2, 3.556534376545218E-4);
    Particle *result = ParticleDownload(gpu);

    for( int i = 0; i < 10; i++ ) {
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].vp[j], result[i].vp[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].xp[j], result[i].xp[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].uf[j], result[i].uf[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].xrhs[j], result[i].xrhs[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].vrhs[j], result[i].vrhs[j] ) << "I: " << i << " J: " << j;
		}

		ASSERT_FLOAT_EQ( expected[i].Tp, result[i].Tp ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tprhs_s, result[i].Tprhs_s ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tprhs_L, result[i].Tprhs_L ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tf, result[i].Tf ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].radius, result[i].radius ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].radrhs, result[i].radrhs ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].qinf, result[i].qinf ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].qstar, result[i].qstar ) << "I: " << i;
	}
}
