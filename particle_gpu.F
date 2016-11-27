      module particle_struct
        use iso_c_binding
        implicit none

        type, bind(c) :: gpu_particle
          integer(c_int) :: pidx, procidx
          real(c_double) :: vp(3), xp(3), uf(3), xrhs(3), vrhs(3)
          real(c_double) :: Tp, Tprhs_s, Tprhs_L, Tf, radius
          real(c_double) :: radrhs, qinf, qstar
        end type gpu_particle
      end module

      module particle_c
#ifdef BUILD_CUDA
      interface
          integer function gpudevices() bind(c,name="gpudevices")
                use iso_c_binding, only: c_int
                integer(c_int) :: count
          end function

          type(c_ptr) function newgpu(count,h,w,d,z)
     +      bind(c,name="NewGPU")
                use iso_c_binding, only: c_ptr, c_int
                integer(c_int), VALUE, intent(in)      :: count
                integer(c_int), VALUE, intent(in)      :: h
                integer(c_int), VALUE, intent(in)      :: w
                integer(c_int), VALUE, intent(in)      :: d
                integer(c_int), VALUE, intent(in)      :: z
          end function

          subroutine gpuwrite(gpu) bind(c,name="ParticleWrite")
            use particle_struct
            use iso_c_binding, only: c_ptr
            type(c_ptr), VALUE        :: gpu
          end subroutine

          subroutine gpuadd(gpu, position, input)
     +      bind(c,name="ParticleAdd")
                use particle_struct
                use iso_c_binding, only: c_ptr, c_int
                type(c_ptr), VALUE        :: gpu
                integer(c_int), VALUE     :: position
                type(gpu_particle)        :: input
          end subroutine

          type(gpu_particle) function gpuget(gpu, position)
     +      bind(c,name="ParticleGet")
                use particle_struct
                use iso_c_binding, only: c_ptr, c_int
                type(c_ptr), VALUE        :: gpu
                integer(c_int), VALUE     :: position
          end function

          subroutine gpuupload(gpu) bind(c,name="ParticleUpload")
                use iso_c_binding, only: c_ptr
                type(c_ptr), VALUE, intent(in)  :: gpu
          end subroutine

          subroutine gpuinterpolate(gpu,dx,dy,nnz,z,zz,ox,oy,oz,uext,
     +      vext,wext,text,t2ext) bind(c,name="ParticleInterpolate")
              use iso_c_binding, only: c_ptr, c_int, c_double
              type(c_ptr), VALUE, intent(in)      :: gpu
              real(c_double), VALUE, intent(in)   :: dx
              real(c_double), VALUE, intent(in)   :: dy
              integer(c_int), VALUE, intent(in)   :: nnz
              real(c_double), intent(in), dimension(*)  :: z
              real(c_double), intent(in), dimension(*)  :: zz
              integer(c_int), VALUE, intent(in)   :: ox
              integer(c_int), VALUE, intent(in)   :: oy
              integer(c_int), VALUE, intent(in)   :: oz
              real(c_double), intent(in), dimension(*)  :: uext
              real(c_double), intent(in), dimension(*)  :: vext
              real(c_double), intent(in), dimension(*)  :: wext
              real(c_double), intent(in), dimension(*)  :: text
              real(c_double), intent(in), dimension(*)  :: t2ext
          end subroutine

          subroutine gpustep(gpu, it, istage, dt)
     +      bind(c,name="ParticleStep")
              use iso_c_binding, only: c_ptr, c_int, c_double
              type(c_ptr), VALUE, intent(in)      :: gpu
              integer(c_int), VALUE, intent(in)   :: it
              integer(c_int), VALUE, intent(in)   :: istage
              real(c_double), VALUE, intent(in)   :: dt
          end subroutine

          subroutine gpunonperiodic(gpu, grid_width, delta_viz)
     +      bind(c,name="ParticleUpdateNonPeriodic")
              use iso_c_binding, only: c_ptr, c_double
              type(c_ptr), VALUE, intent(in)      :: gpu
              real(c_double), VALUE, intent(in)   :: grid_width
              real(c_double), VALUE, intent(in)   :: delta_viz
          end subroutine

          subroutine gpuperiodic(gpu, width, height)
     +      bind(c,name="ParticleUpdatePeriodic")
              use iso_c_binding, only: c_ptr, c_double
              type(c_ptr), VALUE, intent(in)      :: gpu
              real(c_double), VALUE, intent(in)   :: width
              real(c_double), VALUE, intent(in)   :: height
          end subroutine

          subroutine gpudownload(gpu)
     +      bind(c,name="ParticleDownloadHost")
              use iso_c_binding, only: c_ptr
              type(c_ptr), VALUE, intent(in)  :: gpu
          end subroutine
      end interface

      contains
            function select_gpu_master(processors) result(id)
                  integer, intent(in)     :: processors
                  integer                 :: id
                  include 'mpif.h'

                  integer :: gpu_count = 0, i = 0, myid = 0, ierr = 0
                  integer, dimension(processors) :: gpus

                  call mpi_comm_rank(mpi_comm_world,myid,ierr)

                  gpu_count = gpudevices()
                  call mpi_gather(gpu_count, 1, mpi_integer, gpus, 1,
     +                  mpi_integer, 0, mpi_comm_world, ierr)

                  if( myid .eq. 0 ) then
                        id = -1
                        do  i=1,processors
                              if( gpus(i) .gt. 0 ) then
                                    id = i-1
                                    exit
                              end if
                        end  do
                  end if
            end function
#endif

      end module