      module pars
        integer :: iti,itmax,imean,ihst,itape
        integer :: itstr,it_his,it_viz,i_viz

        integer :: ismlt,ifree,iradup,
     +   iupwnd,ibuoy,ifilt,itcut,isubs,ibrcl,iocean,method,
     +   idebug,iz_space,ivis0,ifix_dt,new_vis,iDNS,ispray,
     +   icouple,iTcouple,iHcouple,ievap,ifields,ilin

        integer, parameter :: j_recl=4, k8=8
        integer, parameter :: nscl = 2, nvar = (4+nscl)
        integer, parameter :: maxnx=128,maxny=128,maxnz=128
        integer, parameter :: maxnz1=maxnz+1, maxnz2=maxnz+2,
     +   maxnx2=maxnx+2,maxny2=maxny+2

        real :: Uo, Ttop(nscl), Tbot(nscl),qstar(nscl),tsfcc(nscl)
        real :: grav,t00,fcor,zi,zl,xl,yl,zw1,dpdx
        real :: ugcont,vgcont,zo
        integer :: isfc(nscl)

        integer :: nnx, nny, nnz, nxy, ncx, nnxp1, nnyp1, ncy,
     +   nnxp2, nnyp2, nnzp1, ivis, nnzm1, isize, krec,
     +   izs, ize, ixs, ixe, jxs, jxe, kxs, kxe,
     +   mxs, mxe, iss, ise, iys, iye, jys, jye

        character case*3

        integer :: nvel, npre, nhis1, nprt,
     +   nhisp, nvelc, nviz_z, nviz_y, nviz_x, nviz_s,
     +   kfile, jfile, ibcl, ibcu,
     +   igrdr, imach, itn, it_his_nxt, it_viz_nxt
        logical :: mnout, micut, mtape, mhis, msave,mtrans,
     +     l_root, l_debug, msave_v, mviz

        real :: windm,u1xy,v1xy,t1xy(nscl),
     +     t10xy(nscl),au13m,au23m,aut3m(nscl),tsfcm(nscl),
     +     thstar(nscl), eavg(maxnz), tr_tau(0:maxnz),
     +     pxym(0:maxnz1), zi_min
        integer :: izi, iz_min
        real, allocatable ::
     +     wind(:,:), tau13m(:,:), tau23m(:,:),
     +     taut3m(:,:,:), t_grnd(:,:,:)

        real :: u_mn(0:maxnz1), v_mn(0:maxnz1),
     +   w_mn(0:maxnz1), t_mn(0:maxnz1,nscl)

        real :: dzw(0:maxnz2), dzu(0:maxnz2),
     +   dzw_i(0:maxnz2), dzu_i(0:maxnz2)

        real :: t_factor, t_ref, c_rate, t_surf_i

        real :: dfac(maxnz), dsl_z(0:maxnz1),
     +   xksurf, viscon, vise, almin_c,stabmin,
     +   ck,ceps,csmag,stab_c,vis_mean(0:maxnz)

        integer :: nmatch

        real :: zetas(3), gama(3), etas(4), dt_new,
     +   umax,vmax,wmax, wabs, vismax, cfl, tzero, ucfl, vcfl, wcfl

        character*80  path_res, path_sav, path_his, path_prt,
     +   path_hp, path_sav_hp, path_part, path_ran,
     +   path_v, path_c, path_p, path_h, path_sav_v, path_sav_c,
     +   path_sav_p, path_sav_h, path_sav_part,bad_news
        character case_inp*3
        character*80 path_viz_xy, path_viz_xz, path_viz_yz, path_stuf,
     +   path_seed
        integer :: myid, numprocs, i_root, ziloc, myid_newvis, ncpu_s,
     +   ncpu_z, maxp
        integer, allocatable, dimension(:) ::
     +   ix_s, ix_e, jx_s, jx_e,
     +   kx_s, kx_e, mx_s, mx_e,
     +   iy_s, iy_e, jy_s, jy_e,
     +   is_s, is_e, iz_s, iz_e

        integer :: imultistep, substeps
      contains
      end module
