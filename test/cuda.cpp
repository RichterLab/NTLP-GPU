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

TEST( ParticleCUDA, ParticleInit ) {
	GPU *gpu = (GPU*) malloc( sizeof(GPU) );
	ParticleGenerate(gpu, 2, 10, 1080, 300.0, 0.0, 0.251327, 0.0, 0.1256635, 4e-2, 1.63e-4, 4.0e-5, 1e-2);

	std::vector<Particle> expected = ReadParticles( "../test/data/particle_init.dat" );
	ASSERT_EQ( expected.size(), 10 );

	Particle *result = ParticleDownload(gpu);
	for( int i = 0; i < 10; i++ ) {
		for( int k = 0; k < 10; k++ ){
			if(expected[i].pidx != result[k].pidx) continue;
			if(expected[i].procidx != result[k].procidx) continue;

			ASSERT_FLOAT_EQ( expected[i].pidx, result[k].pidx ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].procidx, result[k].procidx ) << "I: " << i;

			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].vp[j], result[k].vp[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].xp[j], result[k].xp[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].uf[j], result[k].uf[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].xrhs[j], result[k].xrhs[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].vrhs[j], result[k].vrhs[j] ) << "I: " << i << " J: " << j;
			}

			ASSERT_FLOAT_EQ( expected[i].Tp, result[k].Tp ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tprhs_s, result[k].Tprhs_s ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tprhs_L, result[k].Tprhs_L ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tf, result[k].Tf ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].radius, result[k].radius ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].radrhs, result[k].radrhs ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].qinf, result[k].qinf ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].qstar, result[k].qstar ) << "I: " << i;
		}
	}
}

TEST( ParticleCUDA, ParticleUpdate ) {
	GPU *input = ParticleRead("../test/data/c_particle_init.dat");
	ASSERT_EQ( input->pCount, 10 );

	GPU *expected = ParticleRead("../test/data/c_particle_expected.dat");
	ASSERT_EQ( expected->pCount, 10 );

	ParticleUpload(input);
	ParticleStep(input, 1, 1, 4.134832649154196E-004);
    Particle *result = ParticleDownload(input);

    for( int i = 0; i < 10; i++ ) {
		for( int k = 0; k < 10; k++ ){
			if(expected->hParticles[i].pidx != result[k].pidx) continue;
			if(expected->hParticles[i].procidx != result[k].procidx) continue;

			EXPECT_FLOAT_EQ( expected->hParticles[i].pidx, result[k].pidx ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].procidx, result[k].procidx ) << "I: " << i;

			for( int j = 0; j < 3; j++ ) {
				EXPECT_FLOAT_EQ( expected->hParticles[i].vp[j], result[k].vp[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				EXPECT_FLOAT_EQ( expected->hParticles[i].xp[j], result[k].xp[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				EXPECT_FLOAT_EQ( expected->hParticles[i].uf[j], result[k].uf[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				EXPECT_FLOAT_EQ( expected->hParticles[i].xrhs[j], result[k].xrhs[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				EXPECT_FLOAT_EQ( expected->hParticles[i].vrhs[j], result[k].vrhs[j] ) << "I: " << i << " J: " << j;
			}

			EXPECT_FLOAT_EQ( expected->hParticles[i].Tp, result[k].Tp ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].Tprhs_s, result[k].Tprhs_s ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].Tprhs_L, result[k].Tprhs_L ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].Tf, result[k].Tf ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].Tf, result[k].Tf ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].radius, result[k].radius ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].radrhs, result[k].radrhs ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].qinf, result[k].qinf ) << "I: " << i;
			EXPECT_FLOAT_EQ( expected->hParticles[i].qstar, result[k].qstar ) << "I: " << i;
		}
	}
}

TEST( ParticleCUDA, ParticleUpdateNonPeriodic ) {
    std::vector<Particle> input = ReadParticles( "../test/data/particle_nonperiodic_input.dat" );
	ASSERT_EQ( input.size(), 10 );

	std::vector<Particle> expected = ReadParticles( "../test/data/particle_nonperiodic_expected.dat" );
	ASSERT_EQ( expected.size(), 10 );

	GPU *gpu = (GPU*) malloc( sizeof(GPU) );
	ParticleInit(gpu, 10, &input[0]);
	ParticleUpdateNonPeriodic(gpu, 0.04, 0.000163);
    Particle *result = ParticleDownload(gpu);

    for( int i = 0; i < 10; i++ ) {
		for( int k = 0; k < 10; k++ ){
			if(expected[i].pidx != result[k].pidx) continue;
			if(expected[i].procidx != result[k].procidx) continue;

			ASSERT_FLOAT_EQ( expected[i].pidx, result[k].pidx ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].procidx, result[k].procidx ) << "I: " << i;

			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].vp[j], result[k].vp[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].xp[j], result[k].xp[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].uf[j], result[k].uf[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].xrhs[j], result[k].xrhs[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].vrhs[j], result[k].vrhs[j] ) << "I: " << i << " J: " << j;
			}

			ASSERT_FLOAT_EQ( expected[i].Tp, result[k].Tp ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tprhs_s, result[k].Tprhs_s ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tprhs_L, result[k].Tprhs_L ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tf, result[k].Tf ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].radius, result[k].radius ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].radrhs, result[k].radrhs ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].qinf, result[k].qinf ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].qstar, result[k].qstar ) << "I: " << i;
		}
	}
}

TEST( ParticleCUDA, ParticleUpdatePeriodic ) {
    std::vector<Particle> input = ReadParticles( "../test/data/particle_periodic_input.dat" );
	ASSERT_EQ( input.size(), 10 );

	std::vector<Particle> expected = ReadParticles( "../test/data/particle_periodic_expected.dat" );
	ASSERT_EQ( expected.size(), 10 );

	GPU *gpu = (GPU*) malloc( sizeof(GPU) );
	ParticleInit(gpu, 10, &input[0]);
	ParticleUpdatePeriodic(gpu, 0.251327, 0.251327);
    Particle *result = ParticleDownload(gpu);

    for( int i = 0; i < 10; i++ ) {
		for( int k = 0; k < 10; k++ ){
			if(expected[i].pidx != result[k].pidx) continue;
			if(expected[i].procidx != result[k].procidx) continue;

			ASSERT_FLOAT_EQ( expected[i].pidx, result[k].pidx ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].procidx, result[k].procidx ) << "I: " << i;

			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].vp[j], result[k].vp[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].xp[j], result[k].xp[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].uf[j], result[k].uf[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].xrhs[j], result[k].xrhs[j] ) << "I: " << i << " J: " << j;
			}
			for( int j = 0; j < 3; j++ ) {
				ASSERT_FLOAT_EQ( expected[i].vrhs[j], result[k].vrhs[j] ) << "I: " << i << " J: " << j;
			}

			ASSERT_FLOAT_EQ( expected[i].Tp, result[k].Tp ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tprhs_s, result[k].Tprhs_s ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tprhs_L, result[k].Tprhs_L ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].Tf, result[k].Tf ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].radius, result[k].radius ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].radrhs, result[k].radrhs ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].qinf, result[k].qinf ) << "I: " << i;
			ASSERT_FLOAT_EQ( expected[i].qstar, result[k].qstar ) << "I: " << i;
		}
	}
}
