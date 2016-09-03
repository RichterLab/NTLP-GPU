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
      end module