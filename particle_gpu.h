#ifndef PARTICLE_H_
#define PARTICLE_H_

struct Particle {
	int pidx, procidx;
	double vp[3], xp[3], uf[3], xrhs[3], vrhs[3];
	double Tp, Tprhs_s, Tprhs_L, Tf, radius, radrhs, qinf, qstar;
};

struct GPU {
	unsigned int pCount;
	Particle* dParticles;
	Particle* hParticles;
};

extern "C" double rand2(int idum, bool reset = false);
extern "C" GPU* NewGPU(const int particles);
extern "C" void ParticleAdd( GPU *gpu, const int position, const Particle* input );
extern "C" void ParticleUpload( GPU *gpu );
extern "C" void ParticleInit( GPU* gpu, const int particles, const Particle* input );
extern "C" void ParticleGenerate( GPU* gpu, const int processors, const int particles, const int seed, const double temperature, const double xmin, const double xmax, const double ymin, const double ymax, const double zl, const double delta_vis, const double radius, const double qinfp );
extern "C" void ParticleStep( GPU* gpu, const int it, const int istage, const double dt );
extern "C" void ParticleUpdateNonPeriodic( GPU *gpu, const double grid_width, const double delta_viz );
extern "C" void ParticleUpdatePeriodic( GPU *gpu, const double grid_width, const double grid_height );
extern "C" Particle* ParticleDownload( GPU* gpu );

#endif // PARTICLE_H_