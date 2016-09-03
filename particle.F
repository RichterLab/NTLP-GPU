      module particles
        implicit none

        integer :: rproc,trproc,tproc,tlproc,lproc,blproc,bproc,brproc
        integer :: pr_r,pl_r,pt_r,pb_r,ptr_r,ptl_r,pbl_r,pbr_r
        integer :: pr_s,pl_s,pt_s,pb_s,ptr_s,ptl_s,pbl_s,pbr_s
        real :: ymin,ymax,zmin,zmax,xmax,xmin
        real, allocatable :: uext(:,:,:), vext(:,:,:), wext(:,:,:)
        real, allocatable :: u_t(:,:,:), v_t(:,:,:), w_t(:,:,:)
        real, allocatable :: Text(:,:,:),T_t(:,:,:)
        real, allocatable :: T2ext(:,:,:),T2_t(:,:,:)
        real, allocatable :: partTsrc(:,:,:),partTsrc_t(:,:,:)
        real, allocatable :: partHsrc(:,:,:),partHsrc_t(:,:,:)
        real, allocatable :: partTEsrc(:,:,:),partTEsrc_t(:,:,:)
        real, allocatable :: partcount_t(:,:,:),partsrc_t(:,:,:,:)
        real, allocatable :: vpsum_t(:,:,:,:),vpsqrsum_t(:,:,:,:)
        real, allocatable :: upwp_t(:,:,:),upwp(:,:,:)
        real, allocatable :: partcount(:,:,:),partsrc(:,:,:,:)
        real, allocatable :: vpsum(:,:,:,:),vpsqrsum(:,:,:,:)
        real, allocatable :: Tpsum(:,:,:),Tpsum_t(:,:,:)
        real, allocatable :: Tpsqrsum(:,:,:),Tpsqrsum_t(:,:,:)
        real, allocatable :: Tfsum(:,:,:),Tfsum_t(:,:,:)
        real, allocatable :: qfsum(:,:,:),qfsum_t(:,:,:)
        real, allocatable :: wpTpsum(:,:,:),wpTpsum_t(:,:,:)
        real, allocatable :: radsum(:,:,:),radsum_t(:,:,:)
        real, allocatable :: rad2sum(:,:,:),rad2sum_t(:,:,:)
        real, allocatable :: mpsum(:,:,:),mpsum_t(:,:,:)
        real, allocatable :: mwsum(:,:,:),mwsum_t(:,:,:)
        real, allocatable :: qstarsum(:,:,:),qstarsum_t(:,:,:)

        integer :: particletype,pad_diff
        integer :: numpart,tnumpart,ngidx
        integer :: iseed
        real :: Rep_avg,part_grav
        real :: vp_init(3),Tp_init,qf_init

        real :: radius_init,radius_mass,rhow,nuf
        real :: CpaCpp,Cpa,Cpp,Pra,Sc,Mw,Ru,rhoa
        real :: m_s,Sal,Ms,delta_vis,Gam,Ion,Os

        !REMEMBER: IF ADDING ANYTHING, MUST UPDATE MPI DATATYPE!
        type :: particle
        integer :: pidx,procidx
        real :: vp(3),xp(3),uf(3),xrhs(3),vrhs(3),Tp,Tprhs_s
        real :: Tprhs_L,Tf,radius,radrhs,qinf,qstar
        type(particle), pointer :: prev,next
        end type particle

        type(particle), pointer :: part,first_particle
      contains
        subroutine particle_setup
        use pars
        implicit none
        include 'mpif.h'

        integer :: blcts(3),types(3)
        integer :: ierr
        real :: pi
        integer(kind=MPI_ADDRESS_KIND) :: extent,lb
        integer(kind=MPI_ADDRESS_KIND) :: extent2,lb2,displs(3)
        integer :: num_reals

        !First set up the neighbors for the interpolation stage:
        call assign_nbrs

        !Also assign the x,y,z max and mins to track particles leaving
        call set_bounds

        pi   = 4.0*atan(1.0)
        m_s = Sal/1000.0*4.0/3.0*pi*radius_mass**3*rhow

        if (myid==0) write(*,*) 'Init Particle radius = ',radius_init
        if (myid==0) write(*,*) 'Particle Salinity = ',Sal
        if (myid==0) write(*,*) 'Particle gravity = ',part_grav
        if (myid==0) write(*,*) 'nuf = ',nuf
        if (myid==0) write(*,*) 'CpaCpp = ',CpaCpp
        if (myid==0) write(*,*) 'Pra = ',Pra

        !Initialize the linked list of particles:
        nullify(part,first_particle)

        !Set up MPI datatypes for sending particle information
        !MUST UPDATE IF THINGS ARE ADDED/REMOVED FROM PARTICLE STRUCTURE
        num_reals = 5*3+8

        blcts(1:3) = (/2,num_reals,2/)
        displs(1) = 0
        types(1) = mpi_integer
        call mpi_type_get_extent(mpi_integer,lb,extent,ierr)

        !Displace 2*size of mpi_integer (2 integer: pidx,procidx)
        displs(2) = extent*2
        types(2) = mpi_real8
        call mpi_type_get_extent(mpi_real8,lb,extent,ierr)
        !Displace num_reals*size of mpi_real8
        displs(3) = displs(2) + extent*num_reals
        types(3) = mpi_integer8

        !Now define the type:
        call mpi_type_create_struct(3,blcts,displs,types,
     +     particletype,ierr)


        call mpi_type_get_true_extent(particletype,lb2,extent2,ierr)
        call mpi_type_get_extent(particletype,lb2,extent,ierr)
        if (extent .NE. sizeof(part) ) then
            if (myid==0) then
            write(*,*) 'WARNING: extent of particletype not equal
     +                  to sizeof(part):'
            write(*,*) 'sizeof(part) = ', sizeof(part)
            write(*,*) 'mpi_type_get_true_extent(particletype) = ',
     +        extent2
            write(*,*) 'mpi_type_get_extent(particletype) = ',extent
            end if
        end if

        !Need to compute any padding which may exist in particle struct:
        pad_diff = extent-extent2
        if (myid==0) then
          write(*,*) 'mpi_get_extent = ',extent
          write(*,*) 'mpi_get_true_extent = ',extent2
          write(*,*) 'sizeof(part) = ',sizeof(part)
          write(*,*) 'DIFF = ',pad_diff
        end if
        if (pad_diff .LT. 0) then
          write(*,*) 'WARNING: mpi_get_extent - mpi_get_true_extent
     +                LT 0!'
          call mpi_finalize(ierr)
          stop
        end if

        if (myid==0) then
          write(*,*) 'huge(tnumpart) = ',huge(tnumpart)
          write(*,*) 'huge(part%pidx) = ',huge(part%pidx)
        end if

        call mpi_type_commit(particletype,ierr)
        end subroutine particle_setup

        subroutine particle_init
          use pars
          use con_data
          implicit none
          include 'mpif.h'
          integer :: values(8)
          integer :: idx
          real :: xv,yv,zv,ran2,deltaz
          real :: maxx,maxy,maxz

          !Create the seed for the random number generator:
          call date_and_time(VALUES=values)
          iseed = -(myid+values(8)+values(7)+values(6))

          !Initialize ngidx, the particle global index
          ngidx = 1

          !For the channel case, set the total number of particles:
          deltaz = zmax-zmin

          numpart = tnumpart/numprocs
          if (myid == 0) then
            numpart = numpart + MOD(tnumpart,numprocs)
          endif

          !Initialize the linked list of particles:
          nullify(part,first_particle)

          !Now initialize all particles with a random location on that processor
          maxx=0.0
          maxy=0.0
          maxz=0.0
          do idx=1,numpart
            xv = ran2(iseed)*(xmax-xmin) + xmin
            yv = ran2(iseed)*(ymax-ymin) + ymin
            zv = ran2(iseed)*(zl-2.0*delta_vis) + delta_vis
            call particle_create((/xv,yv,zv/),vp_init,Tp_init,qf_init,
     +        idx)
          end do

          partTsrc = 0.0
          partTsrc_t = 0.0
          partHsrc = 0.0
          partHsrc_t = 0.0
          partTEsrc = 0.0
          partTEsrc_t = 0.0
        end subroutine particle_init

        subroutine particle_create(xp,vp,Tp,qinfp,idx)
          !Currently ignoring Tp and qinfp
          use pars
          implicit none

          real :: xp(3),vp(3),Tp,qinfp
          integer :: idx

          if (.NOT. associated(first_particle)) then
            allocate(first_particle)
            part => first_particle
            nullify(part%next,part%prev)
          else
            !Add to beginning of list since it's more convenient
            part => first_particle
            allocate(part%prev)
            first_particle => part%prev
            part%prev%next => part
            part => first_particle
            nullify(part%prev)
          end if

          part%xp(1:3) = xp(1:3)
          part%vp(1:3) = vp(1:3)
          part%Tp = Tp
          part%radius = radius_init
          part%uf(1:3) = 0.0
          part%qinf = qinfp
          part%xrhs(1:3) = 0.0
          part%vrhs(1:3) = 0.0
          part%Tprhs_s = 0.0
          part%Tprhs_L = 0.0
          part%radrhs = 0.0
          part%pidx = idx
          part%procidx = myid
        end subroutine particle_create

        subroutine particle_destroy
          implicit none

          type(particle), pointer :: tmp

          !Is it the first and last in the list?
          if (associated(part,first_particle) .AND.
     +      (.NOT. associated(part%next)) ) then
              nullify(first_particle)
              deallocate(part)
          else
            !Is it the first particle?
            if (associated(part,first_particle)) then
              first_particle => part%next
              part => first_particle
              deallocate(part%prev)
            !Is it the last particle?
            elseif (.NOT. associated(part%next)) then
              nullify(part%prev%next)
              deallocate(part)
            else
              tmp => part
              part => part%next
              tmp%prev%next => tmp%next
              tmp%next%prev => tmp%prev
              deallocate(tmp)
            end if
          end if
        end subroutine particle_destroy

        subroutine particle_stats(ipt,jpt,kpt)
          use pars
          implicit none
          integer :: i,ipt,jpt,kpt
          real :: rhop,pi

          pi   = 4.0*atan(1.0)
          rhop = (m_s+4.0/3.0*pi*part%radius**3*rhow)/
     +             (4.0/3.0*pi*part%radius**3)


          !Takes in ipt,jpt,kpt as the node to the "bottom left" of the
          !particle
          !(i.e. the node in the negative direction for x,y,z)
          !and computes quantities needed to get particle statistics
          partcount_t(kpt,jpt,ipt) = partcount_t(kpt,jpt,ipt) + 1.0

          !Get su mean, mean-squared of particle velocities at each
          !level
          upwp_t(kpt,jpt,ipt) = upwp_t(kpt,jpt,ipt) +
     +                 part%vp(1)*part%vp(3)

          do i = 1,3
            vpsum_t(kpt,jpt,ipt,i) = vpsum_t(kpt,jpt,ipt,i) + part%vp(i)
            vpsqrsum_t(kpt,jpt,ipt,i)=vpsqrsum_t(kpt,jpt,ipt,i)+
     +                 part%vp(i)**2
          end do

          Tpsum_t(kpt,jpt,ipt) = Tpsum_t(kpt,jpt,ipt) + part%Tp
          Tpsqrsum_t(kpt,jpt,ipt) = Tpsqrsum_t(kpt,jpt,ipt) + part%Tp**2

          Tfsum_t(kpt,jpt,ipt) = Tfsum_t(kpt,jpt,ipt) + part%Tf
          qfsum_t(kpt,jpt,ipt) = qfsum_t(kpt,jpt,ipt) + part%qinf

          wpTpsum_t(kpt,jpt,ipt) = wpTpsum_t(kpt,jpt,ipt) +
     +                 part%Tp*part%vp(3)


          radsum_t(kpt,jpt,ipt) = radsum_t(kpt,jpt,ipt) + part%radius
          rad2sum_t(kpt,jpt,ipt) = rad2sum_t(kpt,jpt,ipt) +
     +                  part%radius**2

          mpsum_t(kpt,jpt,ipt) = mpsum_t(kpt,jpt,ipt) +
     +                 rhop*4.0/3.0*pi*part%radius**3

          mwsum_t(kpt,jpt,ipt) = mwsum_t(kpt,jpt,ipt) +
     +                 rhow*4.0/3.0*pi*part%radius**3

          qstarsum_t(kpt,jpt,ipt) = qstarsum_t(kpt,jpt,ipt) + part%qstar
        end subroutine particle_stats

        subroutine particle_update_rk3(it,istage)
          use pars
          use con_data
          use con_stats
          implicit none
          include 'mpif.h'

          integer :: istage,ierr,it
          real :: g(3)
          real :: denom,dtl,sigma,ttest
          integer :: ix,iy,iz
          real :: Rep,diff(3),diffnorm,corrfac,myRep_avg
          real :: xtmp(3),vtmp(3),Tptmp,radiustmp
          real :: Nup,Shp,rhop,taup_i,estar,einf
          real :: myphip_sum,myphiw_sum,myphiv_sum,Volp
          real :: TfC,TpC,Lv,Eff_C,Eff_S
          real :: t_s,t_f,t_s1,t_f1

          g(1:3) = (/0.0, 0.0, part_grav/)

          !First fill extended velocity field for interpolation
          !t_s = mpi_wtime()
          call fill_ext
          !t_f = mpi_wtime()
          !call mpi_barrier(mpi_comm_world,ierr)
          !if (myid==5) write(*,*) 'time fill_ext:',t_f-t_s

          partcount_t = 0.0
          vpsum_t = 0.0
          upwp_t = 0.0
          vpsqrsum_t = 0.0
          Tpsum_t = 0.0
          Tfsum_t = 0.0
          qfsum_t = 0.0
          radsum_t = 0.0
          rad2sum_t = 0.0
          mpsum_t = 0.0
          mwsum_t = 0.0
          Tpsqrsum_t = 0.0
          wpTpsum_t = 0.0
          myRep_avg = 0.0
          myphip_sum = 0.0
          myphiw_sum = 0.0
          myphiv_sum = 0.0
          qstarsum_t = 0.0

          !t_s = mpi_wtime()

          !Loop over the linked list of particles:
          part => first_particle
          do while (associated(part))
            !First, interpolate to get the fluid velocity part%uf(1:3):
            call uf_interp

            if (it .LE. 1 ) then
              part%vp(1:3) = part%uf
              part%Tp = part%Tf
            endif

            !Now advance the particle and position via RK3
            !(same as velocity)

            !Intermediate Values
            diff(1:3) = part%vp - part%uf
            diffnorm = sqrt(diff(1)**2 + diff(2)**2 + diff(3)**2)
            Rep = 2.0*part%radius*diffnorm/nuf
            Volp = pi2*2.0/3.0*part%radius**3
            rhop = (m_s+Volp*rhow)/Volp
            taup_i = 18.0*rhoa*nuf/rhop/(2.0*part%radius)**2

            myRep_avg = myRep_avg + Rep
            corrfac = (1.0 + 0.15*Rep**(0.687))
            myphip_sum = myphip_sum + Volp*rhop
            myphiw_sum = myphiw_sum + Volp*rhow
            myphiv_sum = myphiv_sum + Volp


            !Compute Nusselt number for particle:
            !Ranz-Marshall relation
            Nup = 2.0 + 0.6*Rep**(1.0/2.0)*Pra**(1.0/3.0)
            Shp = 2.0 + 0.6*Rep**(1.0/2.0)*Sc**(1.0/3.0)


            !Mass Transfer calculations
            TfC = part%Tf-273.15
            einf = 610.94*EXP(17.6257*TfC/(TfC+243.04))
            TpC = part%Tp-273.15
            Lv =(25.0 - 0.02274*26.0)*10.0**5

            Eff_C = 2.0*Mw*Gam/(Ru*rhow*part%radius*part%Tp)
            Eff_S = Ion*Os*m_s*Mw/Ms/(Volp*rhop-m_s)
            estar = einf*exp(Mw*Lv/Ru*(1.0/part%Tf-1.0/part%Tp)
     +                                          +Eff_C-Eff_S)
            part%qstar = Mw/Ru*estar/part%Tp/rhoa


            xtmp(1:3) = part%xp(1:3) + dt*zetas(istage)*part%xrhs(1:3)
            vtmp(1:3) = part%vp(1:3) + dt*zetas(istage)*part%vrhs(1:3)
            Tptmp = part%Tp + dt*zetas(istage)*part%Tprhs_s
            Tptmp = Tptmp + dt*zetas(istage)*part%Tprhs_L
            radiustmp = part%radius + dt*zetas(istage)*part%radrhs

            part%xrhs(1:3) = part%vp(1:3)
            part%vrhs(1:3) = corrfac*taup_i*(part%uf(1:3)-part%vp(1:3))
     +                               - g(1:3)

            if (ievap .EQ. 1) then
              !assumes qinf=rhov/rhoa rather than rhov/rhom
              part%radrhs = Shp/9.0/Sc*rhop/rhow*part%radius*
     +                       taup_i*(part%qinf-part%qstar)

            else
              part%radrhs = 0.0
            end if

            part%Tprhs_s = -Nup/3.0/Pra*CpaCpp*
     +                     rhop/rhow*taup_i*(part%Tp-part%Tf)
            part%Tprhs_L = 3.0*Lv/Cpp/part%radius*part%radrhs

            part%xp(1:3) = xtmp(1:3) + dt*gama(istage)*part%xrhs(1:3)
            part%vp(1:3) = vtmp(1:3) + dt*gama(istage)*part%vrhs(1:3)
            part%Tp = Tptmp + dt*gama(istage)*part%Tprhs_s
            part%Tp = part%Tp + dt*gama(istage)*part%Tprhs_L
            part%radius = radiustmp + dt*gama(istage)*part%radrhs

            part => part%next
          end do



          !t_f1 = mpi_wtime()
          !write(*,*) 'proc,loop time: ',myid,t_f1-t_s
          !call mpi_barrier(mpi_comm_world,ierr)
          !t_f = mpi_wtime()
          !if (myid==5) write(*,*) 'time loop:', t_f-t_s

          !Enforce nonperiodic bcs (either elastic or destroying
          ! particles)
          !t_s = mpi_wtime()
          call particle_bcs_nonperiodic
          !call mpi_barrier(mpi_comm_world,ierr)
          !t_f = mpi_wtime()
          !if (myid==5) write(*,*) 'time bc_non:', t_f - t_s

          !Check to see if particles left processor
          !If they did, remove from one list and add to another
          !t_s = mpi_wtime()
          call particle_exchange
          !call mpi_barrier(mpi_comm_world,ierr)
          !t_f = mpi_wtime()
          !if (myid==5) write(*,*) 'time exchg:', t_f - t_s

          !Now enforce periodic bcs
          !just updates x,y locations if over xl,yl or under 0
          !t_s = mpi_wtime()
          call particle_bcs_periodic
          !call mpi_barrier(mpi_comm_world,ierr)
          !t_f = mpi_wtime()
          !if (myid==5) write(*,*) 'time bc_per:', t_f - t_s



          !Now that particles are in their updated position,
          !compute their contribution to the momentum coupling:
          !t_s = mpi_wtime()
          call particle_coupling_update
          !call mpi_barrier(mpi_comm_world,ierr)
          !t_f = mpi_wtime()
          !if (myid==5) write(*,*) 'time cpl: ', t_f - t_s


          !Finally, now that coupling and statistics arrays are filled,
          !Transpose them back to align with the velocities:
          call ztox_trans(partsrc_t(0:nnz+1,iys:iye,mxs:mxe,1),
     +                partsrc(1:nnx,iys:iye,izs-1:ize+1,1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(partsrc_t(0:nnz+1,iys:iye,mxs:mxe,2),
     +                partsrc(1:nnx,iys:iye,izs-1:ize+1,2),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(partsrc_t(0:nnz+1,iys:iye,mxs:mxe,3),
     +                partsrc(1:nnx,iys:iye,izs-1:ize+1,3),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(partTsrc_t(0:nnz+1,iys:iye,mxs:mxe),
     +                partTsrc(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(partHsrc_t(0:nnz+1,iys:iye,mxs:mxe),
     +                partHsrc(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(partTEsrc_t(0:nnz+1,iys:iye,mxs:mxe),
     +                partTEsrc(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)


          !Try only calling these when the history data is being written
          if(mtrans  .and. istage .eq. 3) then
          call ztox_trans(upwp_t(0:nnz+1,iys:iye,mxs:mxe),
     +                upwp(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(vpsum_t(0:nnz+1,iys:iye,mxs:mxe,1),
     +                vpsum(1:nnx,iys:iye,izs-1:ize+1,1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(vpsum_t(0:nnz+1,iys:iye,mxs:mxe,2),
     +                vpsum(1:nnx,iys:iye,izs-1:ize+1,2),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(vpsum_t(0:nnz+1,iys:iye,mxs:mxe,3),
     +                vpsum(1:nnx,iys:iye,izs-1:ize+1,3),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(vpsqrsum_t(0:nnz+1,iys:iye,mxs:mxe,1),
     +                vpsqrsum(1:nnx,iys:iye,izs-1:ize+1,1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(vpsqrsum_t(0:nnz+1,iys:iye,mxs:mxe,2),
     +                vpsqrsum(1:nnx,iys:iye,izs-1:ize+1,2),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(vpsqrsum_t(0:nnz+1,iys:iye,mxs:mxe,3),
     +                vpsqrsum(1:nnx,iys:iye,izs-1:ize+1,3),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(Tpsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                Tpsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(Tpsqrsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                Tpsqrsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(Tfsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                Tfsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(qfsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                qfsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)
          call ztox_trans(wpTpsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                wpTpsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(partcount_t(0:nnz+1,iys:iye,mxs:mxe),
     +                partcount(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(radsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                radsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(rad2sum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                rad2sum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(mpsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                mpsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(mwsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                mwsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          call ztox_trans(qstarsum_t(0:nnz+1,iys:iye,mxs:mxe),
     +                qstarsum(1:nnx,iys:iye,izs-1:ize+1),nnx,nnz,mxs,
     +                mxe,mx_s,mx_e,iys,iye,izs,ize,iz_s,iz_e,myid,
     +                ncpu_s,numprocs)

          end if


          !t_s = mpi_wtime
          !Get particle count:
          numpart = 0
          part => first_particle
          do while (associated(part))
            numpart = numpart + 1
            part => part%next
          end do
          !call mpi_barrier(mpi_comm_world,ierr)
          !t_f = mpi_wtime()
          !if (myid==5) write(*,*) 'time numpart: ', t_f - t_s

          !t_s = mpi_wtime()
          !Compute total number of particles
          call mpi_allreduce(numpart,tnumpart,1,mpi_integer,mpi_sum,
     +                   mpi_comm_world,ierr)
          !Compute average particle Reynolds number
          call mpi_allreduce(myRep_avg,Rep_avg,1,mpi_real8,mpi_sum,
     +                   mpi_comm_world,ierr)

          Rep_avg = Rep_avg/tnumpart

          call mpi_allreduce(myphip_sum,phip,1,mpi_real8,mpi_sum,
     +                   mpi_comm_world,ierr)

          call mpi_allreduce(myphiw_sum,phiw,1,mpi_real8,mpi_sum,
     +                   mpi_comm_world,ierr)

          call mpi_allreduce(myphiv_sum,phiv,1,mpi_real8,mpi_sum,
     +                   mpi_comm_world,ierr)


          phip = phip/xl/yl/zl/rhoa
          phiw = phiw/xl/yl/zl/rhoa
          phiv = phiv/xl/yl/zl

          !call mpi_barrier(mpi_comm_world,ierr)
          !t_f = mpi_wtime()
          !if (myid==5) write(*,*) 'time mpi_allreduce: ', t_f - t_s
        end subroutine particle_update_rk3

        subroutine particle_coupling_update
          use pars
          use con_data
          use con_stats
          implicit none
          include 'mpif.h'
          real :: wtx,wty,wtz,wtt,dV
          real :: rhop,taup_i,partmass
          real :: xv,yv,zv
          real :: ctbuf_s(nnz+2,1:iye-iys+2,6)
          real :: cbbuf_r(nnz+2,1:iye-iys+2,6)
          real :: crbuf_s(nnz+2,1:mxe-mxs+1,6)
          real :: clbuf_r(nnz+2,1:mxe-mxs+1,6)
          integer :: i,j,k,ncount,ipt,jpt,kpt,kwpt
          integer :: istatus(mpi_status_size),ierr
          integer :: ix,iy,iz
          real :: g(3)
          real :: lref,Cvv,Cva,Cpv

          g(1:3) = (/0.0, 0.0, part_grav/)

          partsrc_t = 0.0
          partTsrc_t = 0.0
          partHsrc_t = 0.0
          partTEsrc_t = 0.0

          part => first_particle
          do while (associated(part))

          !First, as done in uf_interp, must find the "leftmost" node
          !of volume where particle belongs:
          !(must repeat since now particle locations have been updated)

          ipt = floor(part%xp(1)/dx) + 1
          jpt = floor(part%xp(2)/dy) + 1
          do iz=0,nnz+1
            if (zz(iz) .GT. part%xp(3)) go to 1299
          end do
 1299  continue
          kpt = iz-1

          !Do again for w:
          do iz = 0,nnz+1
            if (z(iz) .GT. part%xp(3)) go to 1120
          end do
 1120  continue

          kwpt = iz

          !As an aside, use ipt,jpt,kwpt to update the particle numbers
          !at each cell:

          !call inputs are the nodes associated with the volume that
          !each part. lies in
          call particle_stats(ipt,jpt,kwpt)

          !Add contribution to each of the 8 surrounding nodes:
          do i=0,1
          do j=0,1
          do k=0,1

          xv = dx*(i+ipt-1)
          yv = dy*(j+jpt-1)
          zv = zz(k+kpt)

          dV = dx*dy*dzu(kpt+1)

          wtx = (1.0 - abs(part%xp(1)-xv)/dx)
          wty = (1.0 - abs(part%xp(2)-yv)/dy)
          wtz = (1.0 - abs(part%xp(3)-zv)/dzu(kpt+1))
          wtt = wtx*wty*wtz

          rhop = (m_s+pi2*2.0/3.0*part%radius**3*rhow)/
     +                           (pi2*2.0/3.0*part%radius**3)
          partmass = rhop*2.0/3.0*pi2*(part%radius)**3
          taup_i = 18.0*rhoa*nuf/rhop/(2.0*part%radius)**2

          !write(*,'a6,3i') 'node:',ipt+i,jpt+j,kpt+k
          !write(*,'a6,5e15.6') 'Vwt: ',dV,wtx,wty,wtz,wtt

          ix = ipt+i
    !      !Account for periodicity:
    !      if (ix == nnx+1) ix = 1
          iy = jpt+j
          iz = kpt+k

          if (ix .gt. mxe+1) write(*,*) 'proc',myid,'has ix = ',ix
          if (ix .lt. mxs) write(*,*) 'proc',myid,'has ix = ',ix
          if (iy .gt. iye+1) write(*,*) 'proc',myid,'has iy = ',iy
          if (iy .lt. iys) write(*,*) 'proc',myid,'has iy = ',iy
          if (iz .gt. nnz+1) write(*,*) 'proc',myid,'has iz = ',iz
          if (iz .lt. 0) write(*,*) 'proc',myid,'has iz = ',iz

          !Recall to subtract minus g (add g) since momentum is extracted form
          !fluid only through drag term - NOT the gravity term as well
          if (icouple == 1) then
          !drag momentum coupling
          partsrc_t(iz,iy,ix,1:3) =
     +     partsrc_t(iz,iy,ix,1:3) +
     +     partmass/rhoa*(part%vrhs(1:3)+g(1:3))*wtt/dV

          !vapor momentum coupling
          partsrc_t(iz,iy,ix,1:3) =
     +     partsrc_t(iz,iy,ix,1:3) +
     +     rhow/rhoa*pi2*2*part%radius**2*
     +     part%radrhs*part%vp(1:3)*wtt/dV
          endif

          if (iTcouple == 1) then
          partTsrc_t(iz,iy,ix) =
     +     partTsrc_t(iz,iy,ix) +
     +      (part%Tprhs_s*6.0*rhow/rhop/CpaCpp/taup_i*
     +        (pi2/2.0)*part%radius*nuf)*wtt/dV
          endif

          if (iHcouple == 1) then
          partHsrc_t(iz,iy,ix) =     !Brian 8/5/14
     +     partHsrc_t(iz,iy,ix) +  !minus comes later
     +      rhow/rhoa*pi2*2*part%radius**2*part%radrhs*wtt/dV



    !      http://eesc.columbia.edu/courses/ees/climate/SI_Units.pdf
    !      or http://catalog.conveyorspneumatic.com/Asset/FLS%20Specific%20Heat%20Capacities%20of%20Gases.pdf
          Cvv = 1463.0 !J/degkg
          Cpv = 1952.0 !from E. Toolbox -> 1864.0 !J/degkg
          !Cva = 717.0 !J/degkg
          !lref = (25.0 - 0.02274*20.0)*10.0**5
          partTEsrc_t(iz,iy,ix) =
     +     partTEsrc_t(iz,iy,ix) +  !minus comes later
     +         rhow/rhoa*pi2*2*part%radius**2*
     +                  part%radrhs*Cpv/Cpa*part%Tp*wtt/dV -
     +         rhow/rhoa*pi2*2*part%radius**2*
     +                  part%radrhs*Cpv/Cpa*part%Tf*wtt/dV

          endif

          !write(*,'a5,3i,3e15.6') 'bsrc:',ix,iy,iz,partsrc(ix,iy,iz,1:3)

          end do
          end do
          end do

          part => part%next
          end do

          !Now, partsrc and partTsrc have halos on each processor - give these to the rightful owner:
          crbuf_s=0.0;ctbuf_s=0.0
          clbuf_r=0.0;cbbuf_r=0.0

          !First send top:
          !get send buffer ready:
          ctbuf_s(1:nnz+2,1:iye-iys+2,1:3)=partsrc_t(0:nnz+1,
     +                                iys:iye+1,mxe+1,1:3)
          ctbuf_s(1:nnz+2,1:iye-iys+2,4)=partTsrc_t(0:nnz+1,
     +                                iys:iye+1,mxe+1)

          ctbuf_s(1:nnz+2,1:iye-iys+2,5)=partHsrc_t(0:nnz+1,
     +                                iys:iye+1,mxe+1)

          ctbuf_s(1:nnz+2,1:iye-iys+2,6)=partTEsrc_t(0:nnz+1,
     +                                iys:iye+1,mxe+1)

          ncount = 6*(nnz+2)*(iye-iys+2)
          call mpi_sendrecv(ctbuf_s,ncount,mpi_real8,tproc,1,
     +     cbbuf_r,ncount,mpi_real8,bproc,1,mpi_comm_world,istatus,ierr)

          !Now just add the contents of the receive buffer into the entire
          !iys column of this proc:

          partsrc_t(0:nnz+1,iys:iye+1,mxs,1:3) =
     +      partsrc_t(0:nnz+1,iys:iye+1,mxs,1:3) +
     +        cbbuf_r(1:nnz+2,1:iye-iys+2,1:3)

          partTsrc_t(0:nnz+1,iys:iye+1,mxs) =
     +      partTsrc_t(0:nnz+1,iys:iye+1,mxs) +
     +        cbbuf_r(1:nnz+2,1:iye-iys+2,4)

          partHsrc_t(0:nnz+1,iys:iye+1,mxs) =
     +      partHsrc_t(0:nnz+1,iys:iye+1,mxs) +
     +        cbbuf_r(1:nnz+2,1:iye-iys+2,5)


          partTEsrc_t(0:nnz+1,iys:iye+1,mxs) =
     +      partTEsrc_t(0:nnz+1,iys:iye+1,mxs) +
     +        cbbuf_r(1:nnz+2,1:iye-iys+2,6)

          !Now get the right send buffer ready:
          crbuf_s(1:nnz+2,1:mxe-mxs+1,1:3)=
     +        partsrc_t(0:nnz+1,iye+1,mxs:mxe,1:3)

          crbuf_s(1:nnz+2,1:mxe-mxs+1,4)=
     +        partTsrc_t(0:nnz+1,iye+1,mxs:mxe)

          crbuf_s(1:nnz+2,1:mxe-mxs+1,5)=
     +        partHsrc_t(0:nnz+1,iye+1,mxs:mxe)

          crbuf_s(1:nnz+2,1:mxe-mxs+1,6)=
     +        partTEsrc_t(0:nnz+1,iye+1,mxs:mxe)

          !Now send to right:
          ncount = 6*(nnz+2)*(mxe-mxs+1)
          call mpi_sendrecv(crbuf_s,ncount,mpi_real8,rproc,2,
     +     clbuf_r,ncount,mpi_real8,lproc,2,mpi_comm_world,istatus,ierr)

          !And again add the contents to the top/bottom rows of partsrc:
          partsrc_t(0:nnz+1,iys,mxs:mxe,1:3) =
     +           partsrc_t(0:nnz+1,iys,mxs:mxe,1:3) +
     +                clbuf_r(1:nnz+2,1:mxe-mxs+1,1:3)

          partTsrc_t(0:nnz+1,iys,mxs:mxe) =
     +           partTsrc_t(0:nnz+1,iys,mxs:mxe) +
     +                 clbuf_r(1:nnz+2,1:mxe-mxs+1,4)

          partHsrc_t(0:nnz+1,iys,mxs:mxe) =
     +           partHsrc_t(0:nnz+1,iys,mxs:mxe) +
     +                 clbuf_r(1:nnz+2,1:mxe-mxs+1,5)

          partTEsrc_t(0:nnz+1,iys,mxs:mxe) =
     +           partTEsrc_t(0:nnz+1,iys,mxs:mxe) +
     +                 clbuf_r(1:nnz+2,1:mxe-mxs+1,6)
      end subroutine particle_coupling_update

      subroutine particle_bcs_nonperiodic
        use pars
        implicit none
        real :: top,bot

        !Assumes domain goes from [0,xl),[0,yl),[0,zl]
        !Also maintain the number of particles on each proc
        part => first_particle
        do while (associated(part))
          !perfectly elastic collisions on top, bottom walls
          !i.e. location is reflected, w-velocity is negated

          top = zl - delta_vis!part%radius
          bot = 0.0 + delta_vis!part%radius

          if (part%xp(3) .GT. top) then
            part%xp(3) = top - (part%xp(3)-top)
            part%vp(3) = -part%vp(3)
            part => part%next
          elseif (part%xp(3) .LT. bot) then
            part%xp(3) = bot + (bot-part%xp(3))
            part%vp(3) = -part%vp(3)
            part => part%next
          else
            part => part%next
          end if
        end do
      end subroutine particle_bcs_nonperiodic

      subroutine particle_bcs_periodic
        use pars
        implicit none

        !Assumes domain goes from [0,xl),[0,yl),[0,zl]
        !Also maintain the number of particles on each proc
        part => first_particle
        do while (associated(part))
          !x,y periodic

          if (part%xp(1) .GT. xl) then
            part%xp(1) = part%xp(1)-xl
          elseif (part%xp(1) .LT. 0) then
            part%xp(1) = xl + part%xp(1)
          end if

          if (part%xp(2) .GT. yl) then
            part%xp(2) = part%xp(2)-yl
          elseif (part%xp(2) .LT. 0) then
            part%xp(2) = yl + part%xp(2)
          end if

          part => part%next
        end do
      end subroutine particle_bcs_periodic

      subroutine particle_exchange
        use pars
        use con_data
        use con_stats
        implicit none
        include 'mpif.h'

        type(particle), pointer :: tmp
        integer :: idx,psum,csum
        integer :: ir,itr,itop,itl,il,ibl,ib,ibr
        integer :: istatus(mpi_status_size),ierr
        integer :: status_array(mpi_status_size,16),req(16)
        type(particle), allocatable :: rbuf_s(:),trbuf_s(:)
        type(particle), allocatable :: tbuf_s(:),tlbuf_s(:)
        type(particle), allocatable :: lbuf_s(:),blbuf_s(:)
        type(particle), allocatable :: bbuf_s(:),brbuf_s(:)
        type(particle), allocatable :: rbuf_r(:),trbuf_r(:)
        type(particle), allocatable :: tbuf_r(:),tlbuf_r(:)
        type(particle), allocatable :: lbuf_r(:),blbuf_r(:)
        type(particle), allocatable :: bbuf_r(:),brbuf_r(:)
        type(particle), allocatable :: totalbuf(:)

        !Zero out the counters for how many particles to send each dir.
        pr_s=0;ptr_s=0;pt_s=0;ptl_s=0;pl_s=0;pbl_s=0;pb_s=0;pbr_s=0

        !As soon as the location is updated, must check to see if it
        !left the proc:
        !May be a better way of doing this, but it seems most reasonable
        part => first_particle
        do while (associated(part))

          !First get numbers being sent to all sides:
          if (part%xp(2) .GT. ymax) then
              if (part%xp(1) .GT. xmax) then !top right
                ptr_s = ptr_s + 1
              elseif (part%xp(1) .LT. xmin) then !bottom right
                pbr_s = pbr_s + 1
              else  !right
                pr_s = pr_s + 1
              end if
          elseif (part%xp(2) .LT. ymin) then
              if (part%xp(1) .GT. xmax) then !top left
                ptl_s = ptl_s + 1
              else if (part%xp(1) .LT. xmin) then !bottom left
                pbl_s = pbl_s + 1
              else  !left
                pl_s = pl_s + 1
              end if
          elseif ( (part%xp(1) .GT. xmax) .AND.
     +           (part%xp(2) .LT. ymax) .AND.
     +           (part%xp(2) .GT. ymin) ) then !top
              pt_s = pt_s + 1
          elseif ( (part%xp(1) .LT. xmin) .AND.
     +           (part%xp(2) .LT. ymax) .AND.
     +           (part%xp(2) .GT. ymin) ) then !bottom
              pb_s = pb_s + 1
          end if

          part => part%next
        end do

        !Now allocate the send buffers based on these counts:
        allocate(rbuf_s(pr_s),trbuf_s(ptr_s),tbuf_s(pt_s),
     +      tlbuf_s(ptl_s))
        allocate(lbuf_s(pl_s),blbuf_s(pbl_s),bbuf_s(pb_s),
     +      brbuf_s(pbr_s))

        !Now loop back through the particles and fill the buffers:
        !NOTE: If it finds one, add it to buffer and REMOVE from list
        ir=1;itr=1;itop=1;itl=1;il=1;ibl=1;ib=1;ibr=1

        part => first_particle
        do while (associated(part))

          if (part%xp(2) .GT. ymax) then
              if (part%xp(1) .GT. xmax) then !top right
  !               write(*,*) 'Proc',myid,'about to send part to
  !     +proc',trproc,'TOP RIGHT'
                trbuf_s(itr) = part
                call particle_destroy
                itr = itr + 1
              elseif (part%xp(1) .LT. xmin) then !bottom right
  !               write(*,*) 'Proc',myid,'about to send part to
  !     +proc',brproc,'BOTTOM RIGHT'
                brbuf_s(ibr) = part
                call particle_destroy
                ibr = ibr + 1
              else   !right
  !               write(*,*) 'Proc',myid,'about to send part to
  !     +proc',rproc,'RIGHT'
                rbuf_s(ir) = part
                call particle_destroy
                ir = ir + 1
              end if
          elseif (part%xp(2) .LT. ymin) then
              if (part%xp(1) .GT. xmax) then !top left
  !               write(*,*) 'Proc',myid,'about to send part to
  !     +proc',tlproc,'TOP LEFT'
                tlbuf_s(itl) = part
                call particle_destroy
                itl = itl + 1
              else if (part%xp(1) .LT. xmin) then !bottom left
  !               write(*,*) 'Proc',myid,'about to send part to
  !     +proc',blproc,'BOTTOM LEFT'
                blbuf_s(ibl) = part
                call particle_destroy
                ibl = ibl + 1
              else  !left
  !               write(*,*) 'Proc',myid,'about to send part to
  !     +proc',lproc,'LEFT'
                lbuf_s(il) = part
                call particle_destroy
                il = il + 1
              end if
          elseif ( (part%xp(1) .GT. xmax) .AND.
     +           (part%xp(2) .LT. ymax) .AND.
     +           (part%xp(2) .GT. ymin) ) then !top
  !               write(*,*) 'Proc',myid,'about to send part to
  !     +proc',tproc,'TOP'
              tbuf_s(itop) = part
              call particle_destroy
              itop = itop + 1
          elseif ( (part%xp(1) .LT. xmin) .AND.
     +           (part%xp(2) .LT. ymax) .AND.
     +           (part%xp(2) .GT. ymin) ) then !bottom
  !               write(*,*) 'Proc',myid,'about to send part to
  !     +proc',bproc,'BOTTOM'
              bbuf_s(ib) = part
              call particle_destroy
              ib = ib + 1
          else
          part => part%next
          end if

        end do

        !Now everyone exchanges the counts with all neighbors:
        !Left/right:
        call MPI_Sendrecv(pr_s,1,mpi_integer,rproc,3,
     +        pl_r,1,mpi_integer,lproc,3,mpi_comm_world,istatus,ierr)

        call MPI_Sendrecv(pl_s,1,mpi_integer,lproc,4,
     +        pr_r,1,mpi_integer,rproc,4,mpi_comm_world,istatus,ierr)

        !Top/bottom:
        call MPI_Sendrecv(pt_s,1,mpi_integer,tproc,5,
     +        pb_r,1,mpi_integer,bproc,5,mpi_comm_world,istatus,ierr)

        call MPI_Sendrecv(pb_s,1,mpi_integer,bproc,6,
     +        pt_r,1,mpi_integer,tproc,6,mpi_comm_world,istatus,ierr)

        !Top right/bottom left:
        call MPI_Sendrecv(ptr_s,1,mpi_integer,trproc,7,
     +        pbl_r,1,mpi_integer,blproc,7,
     +        mpi_comm_world,istatus,ierr)

        call MPI_Sendrecv(pbl_s,1,mpi_integer,blproc,8,
     +        ptr_r,1,mpi_integer,trproc,8,
     +        mpi_comm_world,istatus,ierr)

        !Top left/bottom right:
        call MPI_Sendrecv(ptl_s,1,mpi_integer,tlproc,9,
     +        pbr_r,1,mpi_integer,brproc,9,
     +        mpi_comm_world,istatus,ierr)

        call MPI_Sendrecv(pbr_s,1,mpi_integer,brproc,10,
     +         ptl_r,1,mpi_integer,tlproc,10,
     +         mpi_comm_world,istatus,ierr)

        !Now everyone has the number of particles arriving from every
        ! neighbor
        !If the count is greater than zero, exchange:

        !Allocate room to receive from each side
        allocate(rbuf_r(pr_r),trbuf_r(ptr_r),tbuf_r(pt_r),
     +          tlbuf_r(ptl_r))
        allocate(lbuf_r(pl_r),blbuf_r(pbl_r),bbuf_r(pb_r),
     +          brbuf_r(pbr_r))

        !Send to right:
        if (pr_s .GT. 0) then
        call mpi_isend(rbuf_s,pr_s,particletype,rproc,11,
     +               mpi_comm_world,req(1),ierr)
        else
        req(1) = mpi_request_null
        end if

        !Receive from left:
        if (pl_r .GT. 0) then
        call mpi_irecv(lbuf_r,pl_r,particletype,lproc,11,
     +               mpi_comm_world,req(2),ierr)
        else
        req(2) = mpi_request_null
        end if

        !Send to left:
        if (pl_s .GT. 0) then
        call mpi_isend(lbuf_s,pl_s,particletype,lproc,12,
     +               mpi_comm_world,req(3),ierr)
        else
        req(3) = mpi_request_null
        end if

        !Receive from right:
        if (pr_r .GT. 0) then
        call mpi_irecv(rbuf_r,pr_r,particletype,rproc,12,
     +               mpi_comm_world,req(4),ierr)
        else
        req(4) = mpi_request_null
        end if

        !Send to top:
        if (pt_s .GT. 0) then
        call mpi_isend(tbuf_s,pt_s,particletype,tproc,13,
     +                mpi_comm_world,req(5),ierr)
        else
        req(5) = mpi_request_null
        end if

        !Receive from bottom:
        if (pb_r .GT. 0) then
        call mpi_irecv(bbuf_r,pb_r,particletype,bproc,13,
     +                mpi_comm_world,req(6),ierr)
        else
        req(6) = mpi_request_null
        end if

        !Send to bottom:
        if (pb_s .GT. 0) then
        call mpi_isend(bbuf_s,pb_s,particletype,bproc,14,
     +                mpi_comm_world,req(7),ierr)
        else
        req(7) = mpi_request_null
        end if

        !Recieve from top:
        if (pt_r .GT. 0) then
        call mpi_irecv(tbuf_r,pt_r,particletype,tproc,14,
     +                mpi_comm_world,req(8),ierr)
        else
        req(8) = mpi_request_null
        end if

        !Send to top right:
        if (ptr_s .GT. 0) then
        call mpi_isend(trbuf_s,ptr_s,particletype,trproc,15,
     +                mpi_comm_world,req(9),ierr)
        else
        req(9) = mpi_request_null
        end if

        !Receive from bottom left:
        if (pbl_r .GT. 0) then
        call mpi_irecv(blbuf_r,pbl_r,particletype,blproc,15,
     +                mpi_comm_world,req(10),ierr)
        else
        req(10) = mpi_request_null
        end if

        !Send to bottom left:
        if (pbl_s .GT. 0) then
        call mpi_isend(blbuf_s,pbl_s,particletype,blproc,16,
     +                mpi_comm_world,req(11),ierr)
        else
        req(11) = mpi_request_null
        end if

        !Receive from top right:
        if (ptr_r .GT. 0) then
        call mpi_irecv(trbuf_r,ptr_r,particletype,trproc,16,
     +                mpi_comm_world,req(12),ierr)
        else
        req(12) = mpi_request_null
        end if

        !Send to top left:
        if (ptl_s .GT. 0) then
        call mpi_isend(tlbuf_s,ptl_s,particletype,tlproc,17,
     +                mpi_comm_world,req(13),ierr)
        else
        req(13) = mpi_request_null
        end if

        !Receive from bottom right:
        if (pbr_r .GT. 0) then
        call mpi_irecv(brbuf_r,pbr_r,particletype,brproc,17,
     +                mpi_comm_world,req(14),ierr)
        else
        req(14) = mpi_request_null
        end if

        !Send to bottom right:
        if (pbr_s .GT. 0) then
        call mpi_isend(brbuf_s,pbr_s,particletype,brproc,18,
     +                mpi_comm_world,req(15),ierr)
        else
        req(15) = mpi_request_null
        end if

        !Receive from top left:
        if (ptl_r .GT. 0) then
        call mpi_irecv(tlbuf_r,ptl_r,particletype,tlproc,18,
     +                mpi_comm_world,req(16),ierr)
        else
        req(16) = mpi_request_null
        end if

        call mpi_waitall(16,req,status_array,ierr)

        !Now add incoming particles to linked list:
        !NOTE: add them to beginning since it's easiest to access (first_particle)

        !Form one large buffer to loop through and add:
        psum = pr_r+ptr_r+pt_r+ptl_r+pl_r+pbl_r+pb_r+pbr_r
        csum = 0
        allocate(totalbuf(psum))
        if (pr_r .GT. 0) then
          totalbuf(1:pr_r) = rbuf_r(1:pr_r)
          csum = csum + pr_r
        end if
        if (ptr_r .GT. 0) then
          totalbuf(csum+1:csum+ptr_r) = trbuf_r(1:ptr_r)
          csum = csum + ptr_r
        end if
        if (pt_r .GT. 0) then
          totalbuf(csum+1:csum+pt_r) = tbuf_r(1:pt_r)
          csum = csum + pt_r
        end if
        if (ptl_r .GT. 0) then
          totalbuf(csum+1:csum+ptl_r) = tlbuf_r(1:ptl_r)
          csum = csum + ptl_r
        end if
        if (pl_r .GT. 0) then
          totalbuf(csum+1:csum+pl_r) = lbuf_r(1:pl_r)
          csum = csum + pl_r
        end if
        if (pbl_r .GT. 0) then
          totalbuf(csum+1:csum+pbl_r) = blbuf_r(1:pbl_r)
          csum = csum + pbl_r
        end if
        if (pb_r .GT. 0) then
          totalbuf(csum+1:csum+pb_r) = bbuf_r(1:pb_r)
          csum = csum + pb_r
        end if
        if (pbr_r .GT. 0) then
          totalbuf(csum+1:csum+pbr_r) = brbuf_r(1:pbr_r)
          csum = csum + pbr_r
        end if

        do idx = 1,psum
          if (.NOT. associated(first_particle)) then
            allocate(first_particle)
            first_particle = totalbuf(idx)
            nullify(first_particle%next,first_particle%prev)
          else
            allocate(first_particle%prev)
            tmp => first_particle%prev
            tmp = totalbuf(idx)
            tmp%next => first_particle
            nullify(tmp%prev)
            first_particle => tmp
            nullify(tmp)
          end if
        end do

        deallocate(rbuf_s,trbuf_s,tbuf_s,tlbuf_s)
        deallocate(lbuf_s,blbuf_s,bbuf_s,brbuf_s)
        deallocate(rbuf_r,trbuf_r,tbuf_r,tlbuf_r)
        deallocate(lbuf_r,blbuf_r,bbuf_r,brbuf_r)
        deallocate(totalbuf)
      end subroutine particle_exchange
      function ran2(idum)
        integer :: idum,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
        real :: ran2,AM,EPS,RNMX
        PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1,
     +     IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,
     +     IR2=3791,NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
        INTEGER :: idum2,j,k,iv(NTAB),iy
        SAVE iv,iy,idum2
        DATA idum2/123456789/,iv/NTAB*0/,iy/0/

        if (idum .le. 0) then
            idum=max(-idum,1)
            idum2 = idum
            do j = NTAB+8,1,-1
              k=idum/IQ1
              idum=IA1*(idum-k*IQ1)-k*IR1
              if (idum .lt. 0) idum=idum+IM1
              if (j .le. NTAB) iv(j) = idum
            end do
            iy=iv(1)
        end if
        k=idum/IQ1
        idum=IA1*(idum-k*IQ1)-k*IR1
        if (idum .lt. 0) idum=idum+IM1
        k=idum2/IQ2
        idum2=IA2*(idum2-k*IQ2)-k*IR2
        if (idum2 .lt. 0) idum2=idum2+IM2
        j = 1+iy/NDIV
        iy = iv(j) - idum2
        iv(j) = idum
        if (iy .lt. 1) iy = iy+IMM1
        ran2=min(AM*iy,RNMX)
        return
        end function ran2
      end module