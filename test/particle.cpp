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

void CompareParticle(Particle* actual, Particle* expected){
	if(actual->pidx != expected->pidx) return;
	if(actual->procidx != expected->procidx) return;

	for( int j = 0; j < 3; j++ ) {
		ASSERT_FLOAT_EQ( actual->vp[j], expected->vp[j] ) << " J: " << j;
	}
	for( int j = 0; j < 3; j++ ) {
		ASSERT_FLOAT_EQ( actual->xp[j], expected->xp[j] ) << " J: " << j;
	}
	for( int j = 0; j < 3; j++ ) {
		ASSERT_FLOAT_EQ( actual->uf[j], expected->uf[j] ) << " J: " << j;
	}
	for( int j = 0; j < 3; j++ ) {
		ASSERT_FLOAT_EQ( actual->xrhs[j], expected->xrhs[j] ) << " J: " << j;
	}
	for( int j = 0; j < 3; j++ ) {
		ASSERT_FLOAT_EQ( actual->vrhs[j], expected->vrhs[j] ) << " J: " << j;
	}

	ASSERT_FLOAT_EQ( actual->Tp, expected->Tp );
	ASSERT_FLOAT_EQ( actual->Tprhs_s, expected->Tprhs_s );
	ASSERT_FLOAT_EQ( actual->Tprhs_L, expected->Tprhs_L );
	ASSERT_FLOAT_EQ( actual->Tf, expected->Tf );
	ASSERT_FLOAT_EQ( actual->radius, expected->radius );
	ASSERT_FLOAT_EQ( actual->radrhs, expected->radrhs );
	ASSERT_FLOAT_EQ( actual->qinf, expected->qinf );
	ASSERT_FLOAT_EQ( actual->qstar, expected->qstar );
}

// ------------------------------------------------------------------
// Non Periodic Boundary Condition Tests
// ------------------------------------------------------------------

// This test should test that the particle is in the center and should
// not change the particle
TEST( Particle, NonPeriodicCenter ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0.25 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 1.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdateNonPeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 1.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is above the top and it
// should invert the Z velocity and set the Z position to (top - (Z-top))
TEST( Particle, NonPeriodicAbove ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0.25 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 1.0}, {0.0, 0.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdateNonPeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, -1.0}, {0.0, 0.0, -0.5}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is below the bottom and it
// should invert the Z velocity and set the Z position to
// (bottom + (bottom + Z))
TEST( Particle, NonPeriodicBelow ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0.25 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, -1.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdateNonPeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 1.0}, {0.0, 0.0, 1.5}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// ------------------------------------------------------------------
// Periodic Boundary Condition Tests
// ------------------------------------------------------------------

// This test should test that the particle is in the center and should
// not change the particle
TEST( Particle, PeriodicCenter ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is negatively out of bounds
// horizontally and should set its X position to Width+X
TEST( Particle, PeriodicNegativeX ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {-0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is negatively out of bounds
// vertical and should set its Y position to Height+Y
TEST( Particle, PeriodicNegativeY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, -0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is negatively out of bounds
// horizontally and vertically and should set its X position to Width+X
// and Y position to Height+Y
TEST( Particle, PeriodicNegativeXY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {-0.25, -0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is positively out of bounds
// horizontally and should set its X position to X-Width
TEST( Particle, PeriodicPositiveX ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {1.0, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is positively out of bounds
// vertically and should set its Y position to Y-Height
TEST( Particle, PeriodicPositiveY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 2.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is positively out of bounds
// horizontally and vertically and should set its X position to X-Width
// and Y position to Y-Height
TEST( Particle, PeriodicPositiveXY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.75, 1.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is negatively out of bounds
// horizontally and positively vertically and should set its X position
// to Width+X and Y position to Y-Height
TEST( Particle, PeriodicNegativeXPositiveY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {-0.25, 1.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// This test should test that the particle is negatively out of bounds
// horizontally and positively vertically and should set its X position
// to Width+X and Y position to Y-Height
TEST( Particle, PeriodicPositiveXNegativeY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0.0 );

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.75, -0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Update Particle
	ParticleUpload(gpu);
	ParticleUpdatePeriodic(gpu);
	ParticleDownload(gpu);

	// Compare Results
	Particle expected = {
		0, 0,
		{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	CompareParticle(&gpu->hParticles[0], &expected);

	// Free Data
	free(gpu);
}

// Interpolate
//