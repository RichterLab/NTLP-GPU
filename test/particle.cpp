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
	ASSERT_EQ(actual->pidx, expected->pidx);
	ASSERT_EQ(actual->procidx, expected->procidx);

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

class ParticleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    params.Evaporation = 1;

	params.rhoa = 1.1;
	params.nuf = 1.537e-5;
	params.Pra = 0.715;

	params.rhow = 1000.0;
	params.Gam = 7.28e-2;
	params.Ion = 2.0;
	params.Os = 1.093;
  }

  Parameters params;
};


// ------------------------------------------------------------------
// Non Periodic Boundary Condition Tests
// ------------------------------------------------------------------

// This test should test that the particle is in the center and should
// not change the particle
TEST_F( ParticleTest, NonPeriodicCenter ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 1.0, 0.0, 0.0, 0.25, &params );

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
TEST_F( ParticleTest, NonPeriodicAbove ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 1.0, 0.0, 0.0, 0.25, &params );

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
TEST_F( ParticleTest, NonPeriodicBelow ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 1.0, 0.0, 0.0, 0.25, &params );

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
TEST_F( ParticleTest, PeriodicCenter ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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
TEST_F( ParticleTest, PeriodicNegativeX ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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
TEST_F( ParticleTest, PeriodicNegativeY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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
TEST_F( ParticleTest, PeriodicNegativeXY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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
TEST_F( ParticleTest, PeriodicPositiveX ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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
TEST_F( ParticleTest, PeriodicPositiveY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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
TEST_F( ParticleTest, PeriodicPositiveXY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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
TEST_F( ParticleTest, PeriodicNegativeXPositiveY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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
TEST_F( ParticleTest, PeriodicPositiveXNegativeY ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 0, 0, 0, 0.5, 1.0, 0.0, 0.0, &params );

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

// ------------------------------------------------------------------
// Neighbour Tests
// ------------------------------------------------------------------

TEST( Particle, Neighbours ) {
	// Setup Variables
	double dx = 0.04188783, dy = 0.04188783;
	double xl = 0.251327, yl = 0.251327;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, -0.00005}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};

	// Get Result
	int *result = ParticleFindXYNeighbours(dx, dy, &input);

	// Compare Results
	int expected[12] = {2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7 };
	for( int i = 0; i < 12; i++ ){
		ASSERT_EQ(result[i], expected[i]);
	}
}

// ------------------------------------------------------------------
// Interpolation Tests
// ------------------------------------------------------------------

TEST_F( ParticleTest, InterpolationZZEQZ ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, -0.00005}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZZEQZ.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZEQ1 ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, 0.00010}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZEQ1.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZLTZ ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, -5.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZLTZ.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZEQ2 ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, 0.0016}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZEQ2.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZEQNNZ ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, 0.04003}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZEQNNZ.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZGTNNZ ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, 0.0401}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZGTNNZ.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZEQNNZM1 ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, 0.0399}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZEQNNZM1.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZEQNNZM2 ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, 0.0385}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZEQNNZM2.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZELSE ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, 0.021}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZELSE.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationZELSE16 ) {
	// Create GPU
	GPU *gpu = NewGPU(1, 21, 21, 18, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/16.0, dy = yl/16.0;

	// Setup Particle
	Particle input = {
		0, 0,
		{0.0, 0.0, 0.0}, {xl / 16.0, yl / 16.0, 0.021}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext16.dat", &size);
	double* vext = ReadArray("../test/data/vext16.dat", &size);
	double* wext = ReadArray("../test/data/wext16.dat", &size);
	double* text = ReadArray("../test/data/text16.dat", &size);
	double* qext = ReadArray("../test/data/qext16.dat", &size);
	double* Z = ReadArray("../test/data/Z16.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ16.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationZELSE16.dat");
	CompareParticle(&gpu->hParticles[0], &expected->hParticles[0]);

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, InterpolationMulti ) {
	// Create GPU
	GPU *gpu = NewGPU(2, 21, 21, 18, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/16.0, dy = yl/16.0;

	// Setup Particle
	Particle input2 = {
		2, 0,
		{0.0, 0.0, 0.0}, {2*dx, 2*dx, 0.021}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 1, &input2);

	Particle input = {
		1, 0,
		{0.0, 0.0, 0.0}, {dx, dy, 0.021}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	ParticleAdd(gpu, 0, &input);

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext16-real.dat", &size);
	double* vext = ReadArray("../test/data/vext16-real.dat", &size);
	double* wext = ReadArray("../test/data/wext16-real.dat", &size);
	double* text = ReadArray("../test/data/text16-real.dat", &size);
	double* qext = ReadArray("../test/data/qext16-real.dat", &size);
	double* Z = ReadArray("../test/data/Z16-real.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ16-real.dat", &size);

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleInterpolate(gpu, dx, dy);
	ParticleDownload(gpu);

	// Compare Results
	GPU *expected = ParticleRead("../test/data/InterpolationMulti.dat");
	ASSERT_EQ(gpu->pCount, expected->pCount);

	for( int i = 0; i < gpu->pCount; i++ ){
		for( int j = 0; j < gpu->pCount; j++ ){
			if( gpu->hParticles[i].pidx == expected->hParticles[j].pidx ){
				CompareParticle(&gpu->hParticles[i], &expected->hParticles[j]);
			}
		}
	}

	// Free Data
	free(gpu);
}

// ------------------------------------------------------------------
// Statistics Tests
// ------------------------------------------------------------------

TEST_F( ParticleTest, StatisticCountEvenDistribution ) {
	// Create GPU
	GPU *gpu = NewGPU(8, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Setup Particle
	for( int i = 0; i < size; i++ ){
		Particle p = { 0, 0, {0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, Z[i]}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		ParticleAdd(gpu, i, &p);
	}

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleCalculateStatistics(gpu, dx, dy);

	// Compare Results
	for( int i = 0; i < size; i++ ){
		ASSERT_EQ(gpu->hPartCount[i], 1);
	}

	// Free Data
	free(gpu);
}

TEST_F( ParticleTest, StatisticCountEveryOther ) {
	// Create GPU
	GPU *gpu = NewGPU(8, 11, 11, 8, 0.5, 1.0, 0.0, 0.0, &params );

	// Setup Variables
	double xl = 0.251327, yl = 0.251327;
	double dx = xl/6.0, dy = yl/6.0;

	// Read Fields
	unsigned int size = 0;
	double* uext = ReadArray("../test/data/uext.dat", &size);
	double* vext = ReadArray("../test/data/vext.dat", &size);
	double* wext = ReadArray("../test/data/wext.dat", &size);
	double* text = ReadArray("../test/data/text.dat", &size);
	double* qext = ReadArray("../test/data/qext.dat", &size);
	double* Z = ReadArray("../test/data/Z.dat", &size);
	double* ZZ = ReadArray("../test/data/ZZ.dat", &size);

	// Setup Particle
	int j = 0;
	for( int i = 0; i < size; i++ ){
		if( i != 0 && i % 2 == 0 ){
			j += 2;
		}
		Particle p = { 0, 0, {0.0, 0.0, 0.0}, {xl / 2.0, yl / 2.0, Z[j]}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		ParticleAdd(gpu, i, &p);
	}

	// Update Particle
	ParticleUpload(gpu);
	ParticleFieldSet(gpu, uext, vext, wext, text, qext, Z, ZZ);
	ParticleCalculateStatistics(gpu, dx, dy);

	// Compare Results
	for( int i = 0; i < size; i++ ){
		if( i % 2 == 0 ){
			ASSERT_EQ(gpu->hPartCount[i], 2);
		}else{
			ASSERT_EQ(gpu->hPartCount[i], 0);
		}
	}

	// Free Data
	free(gpu);
}