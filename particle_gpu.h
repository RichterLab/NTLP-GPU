#ifndef PARTICLE_H_
#define PARTICLE_H_

struct Particle {
	int pidx, procidx;
	double vp[3], xp[3], uf[3], xrhs[3], vrhs[3];
	double Tp, Tprhs_s, Tprhs_L, Tf, radius, radrhs, qinf, qstar;
};

extern "C" double rand2(int idum);
extern "C" Particle* CalculateStep( const int it, const int istage, const double dt, const int pcount, Particle* particles );

#endif // PARTICLE_H_