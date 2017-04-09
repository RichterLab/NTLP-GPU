module class_Profiler
  implicit none

  public :: profiler_initialize

  private
  real :: rate

  type, public :: Profiler
#ifdef BUILD_PERFORMANCE_PROFILE
     integer*8 :: tStart, tEnd
#endif
   contains
     procedure :: start => profile_start
     procedure :: finish => profile_end
  end type Profiler
contains
    subroutine profiler_initialize
#ifdef BUILD_PERFORMANCE_PROFILE
        integer*8 :: cr
        call system_clock(count_rate=cr)
        rate = real(cr)
#endif
    end subroutine

    subroutine profile_start(this, thread, block)
        use pars, only: myid
        include 'mpif.h'

        class(Profiler), intent(inout) :: this
        integer, intent(in) :: thread
        logical, intent(in), optional :: block

#ifdef BUILD_PERFORMANCE_PROFILE
        integer :: ierr
        if( (.not. present(block)) .or. block) then
            call mpi_barrier(mpi_comm_world,ierr)
        endif
        if( myid .eq. thread ) then
            call system_clock(this%tStart)
        end if
#endif
    end subroutine

    subroutine profile_end(this, thread, name, block)
        use pars, only: myid
        include 'mpif.h'

        class(Profiler), intent(inout) :: this
        integer, intent(in) :: thread
        character(len = *), intent(in) :: name
        logical, intent(in), optional :: block

#ifdef BUILD_PERFORMANCE_PROFILE
        integer :: ierr
        if( (.not. present(block)) .or. block) then
            call mpi_barrier(mpi_comm_world,ierr)
        endif
        if( myid .eq. thread ) then
            call system_clock(this%tEnd)
            write(*,'(a,f10.8,a)') name,(this%tEnd-this%tStart)/rate,"s"
        end if
#endif
    end subroutine
end module class_Profiler