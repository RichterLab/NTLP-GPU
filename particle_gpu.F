module particle_struct
    use iso_c_binding
    implicit none

    type, bind(c) :: gpu_particle
        integer(c_int) :: pidx, procidx
        real(c_double) :: vp(3), xp(3), uf(3), xrhs(3), vrhs(3)
        real(c_double) :: Tp, Tprhs_s, Tprhs_L, Tf, radius
        real(c_double) :: radrhs, qinf, qstar
    end type gpu_particle

    type, bind(c) :: gpu_parameters
        integer(c_int) :: Evaporation, LinearInterpolation

        real(c_double) :: rhoa, nuf, Cpa, Pra, Sc

        real(c_double) :: rhow, part_grav, Cpp, Mw, Ru, Ms, Sal, Gam, Ion, Os

        real(c_double) :: radius_mass
    end type gpu_parameters
end module

module particle_c
    use iso_c_binding, only: c_ptr
    implicit none

    type(c_ptr) :: gpu
    integer :: gpu_master_rank

    interface
        integer function gpudevices() bind(c,name="gpudevices")
            use iso_c_binding, only: c_int
            integer(c_int) :: count
        end function

        type(c_ptr) function newgpu(count,h,w,d,xl,yl,zl,z,zz,params) bind(c,name="NewGPU")
            use particle_struct, only: gpu_parameters
            use iso_c_binding, only: c_ptr, c_int, c_double
            integer(c_int), VALUE, intent(in)      :: count
            integer(c_int), VALUE, intent(in)      :: h
            integer(c_int), VALUE, intent(in)      :: w
            integer(c_int), VALUE, intent(in)      :: d
            real(c_double), VALUE, intent(in)      :: xl
            real(c_double), VALUE, intent(in)      :: yl
            real(c_double), VALUE, intent(in)      :: zl
            real(c_double), intent(in), dimension(*)    :: z
            real(c_double), intent(in), dimension(*)    :: zz
            type(gpu_parameters)                   :: params
        end function

        subroutine gpucopyfield(gpu,uext,vext,wext,text,qext) bind(c,name="ParticleFieldSet")
            use iso_c_binding, only: c_ptr, c_double
            type(c_ptr), VALUE                          :: gpu
            real(c_double), intent(in), dimension(*)    :: uext
            real(c_double), intent(in), dimension(*)    :: vext
            real(c_double), intent(in), dimension(*)    :: wext
            real(c_double), intent(in), dimension(*)    :: text
            real(c_double), intent(in), dimension(*)    :: qext
        end subroutine

        subroutine gpuwrite(gpu) bind(c,name="ParticleWrite")
            use particle_struct
            use iso_c_binding, only: c_ptr
            type(c_ptr), VALUE        :: gpu
        end subroutine

        subroutine gpuadd(gpu, position, input) bind(c,name="ParticleAdd")
            use particle_struct
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr), VALUE        :: gpu
            integer(c_int), VALUE     :: position
            type(gpu_particle)        :: input
        end subroutine

        type(gpu_particle) function gpuget(gpu, position) bind(c,name="ParticleGet")
            use particle_struct
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr), VALUE        :: gpu
            integer(c_int), VALUE     :: position
        end function

        subroutine gpuupload(gpu) bind(c,name="ParticleUpload")
            use iso_c_binding, only: c_ptr
            type(c_ptr), VALUE, intent(in)  :: gpu
        end subroutine

        subroutine gpuinterpolate(gpu,dx,dy) bind(c,name="ParticleInterpolate")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
            real(c_double), VALUE, intent(in)   :: dx
            real(c_double), VALUE, intent(in)   :: dy
        end subroutine

        subroutine gpustep(gpu, it, istage, dt) bind(c,name="ParticleStep")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
            integer(c_int), VALUE, intent(in)   :: it
            integer(c_int), VALUE, intent(in)   :: istage
            real(c_double), VALUE, intent(in)   :: dt
        end subroutine

        subroutine gpunonperiodic(gpu) bind(c,name="ParticleUpdateNonPeriodic")
            use iso_c_binding, only: c_ptr, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
        end subroutine

        subroutine gpuperiodic(gpu) bind(c,name="ParticleUpdatePeriodic")
            use iso_c_binding, only: c_ptr, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
        end subroutine

        subroutine gpustatistics(gpu,dx,dy,ox,oy,oz,dzw) bind(c,name="ParticleCalculateStatistics")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr), VALUE, intent(in)              :: gpu
            real(c_double), VALUE, intent(in)           :: dx
            real(c_double), VALUE, intent(in)           :: dy
            integer(c_int), VALUE, intent(in)           :: ox
            integer(c_int), VALUE, intent(in)           :: oy
            integer(c_int), VALUE, intent(in)           :: oz
            real(c_double), intent(inout), dimension(*) :: dzw
        end subroutine

        subroutine gpudownload(gpu) bind(c,name="ParticleDownloadHost")
            use iso_c_binding, only: c_ptr
            type(c_ptr), VALUE, intent(in)  :: gpu
        end subroutine

        subroutine gpufillstatistics(gpu,partcount,vpsum,vpsumsq) bind(c,name="ParticleFillStatistics")
            use iso_c_binding, only: c_ptr,c_double
            type(c_ptr), VALUE, intent(in)  :: gpu
            real(c_double), intent(inout), dimension(*) :: partcount
            real(c_double), intent(inout), dimension(*) :: vpsum
            real(c_double), intent(inout), dimension(*) :: vpsumsq
        end subroutine
    end interface
contains
        subroutine select_gpu_master()
            use pars, only: numprocs, myid

            include 'mpif.h'

            integer :: gpu_count, i, ierr
            integer, dimension(numprocs) :: gpus

            gpu_count = gpudevices()
            call mpi_gather(gpu_count,1,mpi_integer,gpus,1,mpi_integer,0,mpi_comm_world,ierr)

            if( myid .eq. 0 ) then
                gpu_master_rank = -1
                do  i=1,numprocs
                    if( gpus(i) .gt. 0 ) then
                        gpu_master_rank = i-1
                        exit
                    end if
                end  do
            end if

            ! Distribute to all processors
            call mpi_bcast(gpu_master_rank, 1, mpi_integer, 0, mpi_comm_world, ierr)
        end subroutine

        subroutine initialize_gpu()
            use pars, only: maxnx,maxny,maxnz,xl,yl,zl,myid,ievap,ilin
            use con_stats, only: z, zz
            use particles
            use particle_struct, only: gpu_parameters

            type(gpu_parameters) :: parameters

            if( myid .eq. gpu_master_rank ) then
                ! Setup Parameters
                parameters%Evaporation = ievap
                parameters%LinearInterpolation = ilin

                parameters%rhoa = rhoa
                parameters%nuf = nuf
                parameters%Cpa = Cpa
                parameters%Pra = Pra
                parameters%Sc = Sc

                parameters%rhow = rhow
                parameters%part_grav = part_grav
                parameters%Cpp = Cpp
                parameters%Mw = Mw
                parameters%Ru = Ru
                parameters%Ms = Ms
                parameters%Sal = Sal
                parameters%Gam = Gam
                parameters%Ion = Ion
                parameters%Os = Os

                parameters%radius_mass = radius_init

                ! Create GPU Instance
                gpu = newgpu(tnumpart,maxnx+5,maxny+5,maxnz+2,xl,yl,zl,z,zz,parameters)
            end if
        end subroutine

        subroutine transfer_particles()
            use pars, only: numprocs, myid
            use particles
            use particle_struct

            include 'mpif.h'

            integer :: ierr, i, iCurrent
            integer, allocatable :: pCounts(:), pDispls(:)
            type(particle), allocatable :: pCurrent(:), pTotal(:)
            type(gpu_particle) :: gpuParticle

            allocate(pCurrent(numpart), pTotal(tnumpart))
            allocate(pCounts(numprocs), pDispls(numprocs))

            iCurrent = 1
            part => first_particle
            do while (associated(part))
                pCurrent(iCurrent) = part
                iCurrent = iCurrent + 1
                part => part%next
            end do
            iCurrent = iCurrent - 1

            call mpi_gather(iCurrent, 1, mpi_integer, pCounts, 1, mpi_integer, gpu_master_rank, mpi_comm_world, ierr)

            if (myid .eq. gpu_master_rank) then
                iCurrent = 0
                do  i=1,numprocs
                    pDispls(i) = iCurrent
                    iCurrent = iCurrent + pCounts(i)
                end do
            end if

            call mpi_gatherv(pCurrent, numpart, particletype, pTotal, pCounts, pDispls, particletype, gpu_master_rank, mpi_comm_world, ierr)

            if (myid .eq. gpu_master_rank) then
                do i = 1,tnumpart
                    gpuParticle%pidx = pTotal(i)%pidx
                    gpuParticle%procidx = pTotal(i)%procidx

                    gpuParticle%vp(1:3) = pTotal(i)%vp(1:3)
                    gpuParticle%xp(1:3) = pTotal(i)%xp(1:3)
                    gpuParticle%uf(1:3) = pTotal(i)%uf(1:3)
                    gpuParticle%xrhs(1:3) = pTotal(i)%xrhs(1:3)
                    gpuParticle%vrhs(1:3) = pTotal(i)%vrhs(1:3)

                    gpuParticle%Tp = pTotal(i)%Tp
                    gpuParticle%Tprhs_s = pTotal(i)%Tprhs_s
                    gpuParticle%Tprhs_L = pTotal(i)%Tprhs_L
                    gpuParticle%Tf = pTotal(i)%Tf
                    gpuParticle%radius = pTotal(i)%radius
                    gpuParticle%radrhs = pTotal(i)%radrhs
                    gpuParticle%qinf = pTotal(i)%qinf
                    gpuParticle%qstar = pTotal(i)%qstar

                    call gpuadd(gpu, i - 1, gpuParticle)
                end do
                call gpuupload(gpu)
            end if
        end subroutine

        subroutine calculate_statistics(stat)
            use pars, only: nnz, nny, nnx, numprocs, myid, dzw
            use con_data, only: dx, dy, nscl
            use particles
            use particle_struct

            include 'mpif.h'

            integer :: iz, istatus, ierr, dim, tdim
            real :: stat(1:nnz,34 + 5*nscl)
            real(8), allocatable :: pCount(:), vSum(:), vSumSQ(:)

            dim = nnz + 2
            tdim = dim * 3
            allocate(pCount(dim), vSum(tdim), vSumSQ(tdim))

            if( myid .eq. gpu_master_rank ) then
                call gpustatistics(gpu,dx,dy,nnz,nny,nnx,dzw)
                call gpufillstatistics(gpu, pCount, vSum, vSumSQ)

                if( myid .ne. 0 ) then
                    call mpi_send(pCount,dim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(vSum,tdim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(vSumSQ,tdim,mpi_real8,0,1,mpi_comm_world,ierr)
                end if
            else
                if( myid .eq. 0 ) then
                    call mpi_recv(pCount,dim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(vSum,tdim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(vSumSQ,tdim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                end if
            end if

            if( myid .eq. 0 ) then
                do iz=1,nnz
                    stat(iz,11) = pCount(iz)
                    stat(iz,12) = vSum((iz-1)*3+1)
                    stat(iz,13) = vSum((iz-1)*3+2)
                    stat(iz,14) = vSum((iz-1)*3+3)
                    stat(iz,15) = vSumSQ((iz-1)*3+1)
                    stat(iz,16) = vSumSQ((iz-1)*3+2)
                    stat(iz,17) = vSumSQ((iz-1)*3+3)
                end do
            end if
        end subroutine

        subroutine setup_particles()
            use pars, only: myid, xl, yl
            use con_data, only: dx, dy
            use particles
            use fields

            integer :: i, j, k

            ! Setup Particles
            part => first_particle
            do while (associated(part))
                part%vp(1:3) = (/0.0, 0.0, 0.0/)
                part%xp(1:3) = (/0.087370281899,0.125141144198,0.018382617541/)
                part%uf(1:3) = (/0.0, 0.0, 0.0/)
                part%xrhs(1:3) = (/0.0, 0.0, 0.0/)
                part%vrhs(1:3) = (/0.0, 0.0, 0.0/)

                part%Tp = 0.0
                part%Tprhs_s = 0.0
                part%Tprhs_L = 0.0
                part%Tf = 0.0
                part%radius = 0.0
                part%radrhs = 0.0
                part%qinf = 0.0
                part%qstar = 0.0

                part => part%next
            end do

            ! Move to master and save
            call transfer_particles
            if (myid .eq. gpu_master_rank) then
                call gpuwrite(gpu)
            end if

            ! Run interpolation
            call fill_ext

            part => first_particle
            do while (associated(part))
                call uf_interp
                part => part%next
            end do

            ! Move to master and save
            call transfer_particles
            if (myid .eq. gpu_master_rank) then
                call gpuwrite(gpu)
            end if

            stop 0
        end subroutine

        subroutine compare_particles()
            use pars, only: numprocs, myid
            use particles
            use particle_struct

            include 'mpif.h'

            integer :: ierr, i, j, k, iCurrent, succeeded, failures
            integer, allocatable :: pCounts(:), pDispls(:)
            type(particle), allocatable :: pCurrent(:), pTotal(:)
            type(gpu_particle) :: gPart

            allocate(pCurrent(numpart), pTotal(tnumpart))
            allocate(pCounts(numprocs), pDispls(numprocs))

            iCurrent = 1
            part => first_particle
            do while (associated(part))
                pCurrent(iCurrent) = part
                iCurrent = iCurrent + 1
                part => part%next
            end do
            iCurrent = iCurrent - 1

            call mpi_gather(iCurrent, 1, mpi_integer, pCounts, 1, mpi_integer, gpu_master_rank, mpi_comm_world, ierr)

            if (myid .eq. gpu_master_rank) then
                iCurrent = 0
                do  i=1,numprocs
                    pDispls(i) = iCurrent
                    iCurrent = iCurrent + pCounts(i)
                end do
            end if

            call mpi_gatherv(pCurrent, numpart, particletype, pTotal, pCounts, pDispls, particletype, gpu_master_rank, mpi_comm_world, ierr)

            if (myid .eq. gpu_master_rank) then
                failures = 0

                call gpudownload(gpu)
                do i = 1,tnumpart
                    gPart = gpuget(gpu, i-1)

9001                format(a10, 3x, 'expected:', T25, f14.6, T50, 'actual:', 1x, f14.6)
9002                format(a10, 1x, i1, 1x, 'expected:', T25, f14.6, T50, 'actual:', 1x, f14.6)

                    do k = 1,tnumpart
                        if (pTotal(k)%pidx .eq. gPart%pidx) then
                            succeeded = 1

                            do j = 1,3
                                if( dabs(pTotal(k)%vp(j)-gPart%vp(j)) .gt. 1e-5 ) then
                                    succeeded = 0
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%xp(j)-gPart%xp(j)) .gt. 1e-5 ) then
                                    succeeded = 0
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%uf(j) - gPart%uf(j)) .gt. 1e-5 ) then
                                    succeeded = 0
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%xrhs(j)-gPart%xrhs(j)) .gt. 1e-5) then
                                    succeeded = 0
                                end if
                            end do

                            do j = 1,3
                                if(dabs(pTotal(k)%vrhs(j)-gPart%vrhs(j)) .gt. 1e-5) then
                                    succeeded = 0
                                end if
                            end do

                            if( dabs(pTotal(k)%Tp - gPart%Tp) .gt. 1e-5 ) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%Tprhs_s-gPart%Tprhs_s) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if( dabs(pTotal(k)%Tprhs_L-gPart%Tprhs_L) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if( dabs(pTotal(k)%Tf - gPart%Tf) .gt. 1e-5 ) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%radius - gPart%radius) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%radrhs-gPart%radrhs) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%qinf-gPart%qinf) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%qstar-gPart%qstar) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if( succeeded == 0 ) then
                                failures = failures + 1

                                write(*,*) "Particle:", i, "Processor:", pTotal(k)%procidx, "ID:", pTotal(k)%pidx

                                do j = 1,3
                                    write(*,9002) "vp", j, pTotal(k)%vp(j), gPart%vp(j)
                                end do

                                do j = 1,3
                                    write(*,9002) "xp", j, pTotal(k)%xp(j), gPart%xp(j)
                                end do

                                do j = 1,3
                                    write(*,9002) "uf", j, pTotal(k)%uf(j), gPart%uf(j)
                                end do

                                do j = 1,3
                                    write(*,9002) "xrhs", j, pTotal(k)%xrhs(j), gPart%xrhs(j)
                                end do

                                do j = 1,3
                                    write(*,9002) "vrhs", j, pTotal(k)%vrhs(j), gPart%vrhs(j)
                                end do

                                write(*,9001) "Tp", pTotal(k)%Tp, gPart%Tp
                                write(*,9001) "Tprhs_s", pTotal(k)%Tprhs_s, gPart%Tprhs_s
                                write(*,9001) "Tprhs_L", pTotal(k)%Tprhs_L, gPart%Tprhs_L
                                write(*,9001) "Tf", pTotal(k)%Tf, gPart%Tf
                                write(*,9001) "radius", pTotal(k)%radius, gPart%radius
                                write(*,9001) "radrhs", pTotal(k)%radrhs, gPart%radrhs
                                write(*,9001) "qinf", pTotal(k)%qinf, gPart%qinf
                                write(*,9001) "qstar", pTotal(k)%qstar, gPart%qstar

                                write(*,*) ""
                            end if
                        end if
                    end do
                end do

                write(*,*) "Total Failures: ", failures
            end if
        end subroutine
end module