      module particle_c

#ifdef BUILD_CUDA
      implicit none
      interface
            integer function gpudevices() bind(c,name="gpudevices")
                  use iso_c_binding, only: c_int
                  integer(c_int) :: count
            end function
      end interface

      contains
            function select_gpu_master(processors) result(id)
                  integer, intent(in)     :: processors
                  integer                 :: id
                  include 'mpif.h'

                  integer :: gpu_count = 0, i = 0, myid = 0, ierr = 0
                  integer, allocatable, dimension(:) :: gpus

                  call mpi_comm_rank(mpi_comm_world,myid,ierr)

                  gpu_count = gpudevices()
                  if( myid .eq. 0 ) then
                        allocate( gpus(processors) )
                  end if

                  call mpi_gather(gpu_count, 1, mpi_integer, gpus, 1,
     +                  mpi_integer, 0, mpi_comm_world, ierr)

                  if( myid .eq. 0 ) then
                        id = -1
                        do  i=0,processors
                              if( gpus(i) .gt. 0 ) then
                                    id = i
                                    exit
                              end if
                        end  do
                  end if
            end function
#endif

      end module