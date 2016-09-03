      module con_stats
        use pars
        type con_s
        sequence
        real ::  wwsb(maxnz),engz(0:maxnz1),
     +           engsbz(0:maxnz1),
     +           englez(maxnz),uxym(0:maxnz1),
     +           vxym(0:maxnz1),wxym(0:maxnz1),
     +           txym(0:maxnz1,nscl),divz(0:maxnz1),
     +           utle(maxnz,nscl), utsb(maxnz,nscl),
     +           vtle(maxnz,nscl), vtsb(maxnz,nscl),
     +           wtle(maxnz,nscl), wtsb(0:maxnz,nscl),
     +           wt_tot(maxnz,nscl),
     +           z(0:maxnz1),zz(0:maxnz1),
     +           shrz(maxnz),buyz(maxnz),
     +           triz(maxnz),
     +           uwsb(0:maxnz),vwsb(0:maxnz),
     +           uwle(maxnz),vwle(maxnz),
     +           uw_tot(maxnz),vw_tot(maxnz),
     +           wcube(maxnz), wfour(maxnz),
     +           tcube(maxnz,nscl),
     +           ups(maxnz), vps(maxnz),
     +           wps(maxnz), tps(maxnz,nscl),
     +           t_rprod(maxnz), t_wq(maxnz),
     +           t_wp(maxnz), t_tau(maxnz),
     +           t_tran(maxnz), t_buoy(maxnz),
     +           t_diss(maxnz), t_sprod(maxnz),
     +           zconc(maxnz),
     +           vp1mean(maxnz),vp2mean(maxnz),vp3mean(maxnz),
     +           vp1msqr(maxnz),vp2msqr(maxnz),vp3msqr(maxnz),
     +           m1src(maxnz),m2src(maxnz),m3src(maxnz),
     +           upwpm(maxnz),
     +           Tpmean(maxnz),Tpmsqr(maxnz),
     +           Tfmean(maxnz),qfmean(maxnz),
     +           wpTpm(maxnz),Tpsrc(maxnz),
     +           Tv_tran(maxnz,nscl),Tv_prod(maxnz,nscl),
     +           Tv_diss(maxnz,nscl),
     +           Tv_part1(maxnz,nscl),Tv_part2(maxnz),
     +           radmean(maxnz),rad2mean(maxnz),
     +           mpmean(maxnz),mwmean(maxnz),
     +           phimp(maxnz),phimw(maxnz),
     +           Hpsrc(maxnz),TEpsrc(maxnz),
     +           qstarm(maxnz),
     +           trip(maxnz),TpSq(maxnz),Tpdqp(maxnz)
        real ::  xkn(maxnx),ykn(maxny),
     +           xk(maxnx),yk(maxny),
     +           xks(maxnx2,maxny),wavexy(maxnx2,maxny)
        real ::  ug(maxnz),vg(maxnz),
     +           wls(maxnz),uls(maxnx)
        real ::  udrift,vdrift,
     +           stokesw,stokesa,
     +           stokess,stokes(maxnz1)
        real ::  dtg, dslg, dzg
        real ::  phip,phiw,phiv

        end type con_s
        type(con_s), target :: c_s
        real, pointer ::
     +           wwsb(:), engz(:), engsbz(:),
     +           englez(:), uxym(:), vxym(:), wxym(:),
     +           txym(:,:), divz(:), utle(:,:), utsb(:,:),
     +           vtle(:,:), vtsb(:,:), wtle(:,:), wtsb(:,:),
     +           wt_tot(:,:), z(:), zz(:), shrz(:), buyz(:),
     +           triz(:), uwsb(:), vwsb(:), uwle(:), vwle(:),
     +           uw_tot(:), vw_tot(:), wcube(:), wfour(:),
     +           tcube(:,:), ups(:), vps(:),
     +           wps(:), tps(:,:), t_rprod(:), t_wq(:),
     +           t_wp(:), t_tau(:), t_tran(:), t_buoy(:),
     +           t_diss(:), t_sprod(:),
     +           zconc(:),
     +           vp1mean(:),vp2mean(:),vp3mean(:),
     +           vp1msqr(:),vp2msqr(:),vp3msqr(:),
     +           m1src(:),m2src(:),m3src(:),
     +           upwpm(:),
     +           Tpmean(:),Tpmsqr(:),
     +           Tfmean(:),qfmean(:),
     +           wpTpm(:),Tpsrc(:),
     +           Tv_tran(:,:),Tv_prod(:,:),Tv_diss(:,:),
     +           Tv_part1(:,:),Tv_part2(:),
     +           radmean(:),rad2mean(:),
     +           mpmean(:),mwmean(:),
     +           phimp(:),phimw(:),
     +           Hpsrc(:), TEpsrc(:),
     +           qstarm(:),
     +           trip(:),TpSq(:),Tpdqp(:)
        real, pointer ::
     +           xkn(:), ykn(:), xk(:), yk(:), xks(:,:), wavexy(:,:)
        real, pointer ::
     +           ug(:), vg(:), wls(:), uls(:)
        real, pointer ::
     +           udrift, vdrift, stokesw, stokesa,
     +           stokess, stokes(:)
        real, pointer ::
     +           dtg, dslg, dzg
        real, pointer ::
     +           phip,phiw,phiv
      contains
         subroutine fill_cs
c
c -------------- pointer association for stat arrays and get size
c                of stat arrays isize for history files
c
             isize = 0
             wwsb    => c_s%wwsb     ; isize = isize + size(wwsb)
             engz    => c_s%engz     ; isize = isize + size(engz)
             engsbz  => c_s%engsbz   ; isize = isize + size(engsbz)
             englez  => c_s%englez   ; isize = isize + size(englez)
             uxym    => c_s%uxym     ; isize = isize + size(uxym)
             vxym    => c_s%vxym     ; isize = isize + size(vxym)
             wxym    => c_s%wxym     ; isize = isize + size(wxym)
             txym    => c_s%txym     ; isize = isize + size(txym)
             divz    => c_s%divz     ; isize = isize + size(divz)
             utle    => c_s%utle     ; isize = isize + size(utle)
             utsb    => c_s%utsb     ; isize = isize + size(utsb)
             vtle    => c_s%vtle     ; isize = isize + size(vtle)
             vtsb    => c_s%vtsb     ; isize = isize + size(vtsb)
             wtle    => c_s%wtle     ; isize = isize + size(wtle)
             wtsb    => c_s%wtsb     ; isize = isize + size(wtsb)
             wt_tot  => c_s%wt_tot   ; isize = isize + size(wt_tot)
             z       => c_s%z        ; isize = isize + size(z)
             zz      => c_s%zz       ; isize = isize + size(zz)
             shrz    => c_s%shrz     ; isize = isize + size(shrz)
             buyz    => c_s%buyz     ; isize = isize + size(buyz)
             triz    => c_s%triz     ; isize = isize + size(triz)
             uwsb    => c_s%uwsb     ; isize = isize + size(uwsb)
             vwsb    => c_s%vwsb     ; isize = isize + size(vwsb)
             uwle    => c_s%uwle     ; isize = isize + size(uwle)
             vwle    => c_s%vwle     ; isize = isize + size(vwle)
             uw_tot  => c_s%uw_tot   ; isize = isize + size(uw_tot)
             vw_tot  => c_s%vw_tot   ; isize = isize + size(vw_tot)
             wcube   => c_s%wcube    ; isize = isize + size(wcube)
             wfour   => c_s%wfour    ; isize = isize + size(wfour)
             tcube   => c_s%tcube    ; isize = isize + size(tcube)
             ups     => c_s%ups      ; isize = isize + size(ups)
             vps     => c_s%vps      ; isize = isize + size(vps)
             wps     => c_s%wps      ; isize = isize + size(wps)
             tps     => c_s%tps      ; isize = isize + size(tps)
             t_rprod => c_s%t_rprod  ; isize = isize + size(t_rprod)
             t_wq    => c_s%t_wq     ; isize = isize + size(t_wq)
             t_wp    => c_s%t_wp     ; isize = isize + size(t_wp)
             t_tau   => c_s%t_tau    ; isize = isize + size(t_tau)
             t_tran  => c_s%t_tran   ; isize = isize + size(t_tran)
             t_buoy  => c_s%t_buoy   ; isize = isize + size(t_buoy)
             t_diss  => c_s%t_diss   ; isize = isize + size(t_diss)
             t_sprod => c_s%t_sprod  ; isize = isize + size(t_sprod)
             zconc   => c_s%zconc    ; isize = isize + size(zconc)
             vp1mean  => c_s%vp1mean ; isize = isize + size(vp1mean)
             vp2mean  => c_s%vp2mean ; isize = isize + size(vp2mean)
             vp3mean  => c_s%vp3mean ; isize = isize + size(vp3mean)
             vp1msqr  => c_s%vp1msqr ; isize = isize + size(vp1msqr)
             vp2msqr  => c_s%vp2msqr ; isize = isize + size(vp2msqr)
             vp3msqr  => c_s%vp3msqr ; isize = isize + size(vp3msqr)
             m1src   => c_s%m1src    ; isize = isize + size(m1src)
             m2src   => c_s%m2src    ; isize = isize + size(m2src)
             m3src   => c_s%m3src    ; isize = isize + size(m3src)
             upwpm   => c_s%upwpm    ; isize = isize + size(upwpm)
             Tpmean  => c_s%Tpmean   ; isize = isize + size(Tpmean)
             Tpmsqr  => c_s%Tpmsqr   ; isize = isize + size(Tpmsqr)
             Tfmean  => c_s%Tfmean   ; isize = isize + size(Tfmean)
             qfmean  => c_s%qfmean   ; isize = isize + size(qfmean)
             wpTpm   => c_s%wpTpm    ; isize = isize + size(wpTpm)
             Tpsrc   => c_s%Tpsrc    ; isize = isize + size(Tpsrc)
             Tv_tran => c_s%Tv_tran  ; isize = isize + size(Tv_tran)
             Tv_prod => c_s%Tv_prod  ; isize = isize + size(Tv_prod)
             Tv_diss => c_s%Tv_diss  ; isize = isize + size(Tv_diss)
             Tv_part1 => c_s%Tv_part1  ; isize = isize + size(Tv_part1)
             Tv_part2 => c_s%Tv_part2  ; isize = isize + size(Tv_part2)
             radmean => c_s%radmean  ; isize = isize + size(radmean)
             rad2mean => c_s%rad2mean  ; isize = isize + size(rad2mean)
             mpmean => c_s%mpmean    ; isize = isize + size(mpmean)
             mwmean => c_s%mwmean    ; isize = isize + size(mwmean)
             phimp => c_s%phimp    ; isize = isize + size(phimp)
             phimw => c_s%phimw    ; isize = isize + size(phimw)
             Hpsrc   => c_s%Hpsrc    ; isize = isize + size(Hpsrc)
             TEpsrc  => c_s%TEpsrc  ; isize = isize + size(TEpsrc)
             qstarm => c_s%qstarm  ; isize = isize + size(qstarm)
             trip => c_s%trip; isize = isize + size(trip)
             TpSq => c_s%TpSq; isize = isize + size(TpSq)
             Tpdqp => c_s%Tpdqp; isize = isize + size(Tpdqp)


             xkn     => c_s%xkn
             ykn     => c_s%ykn
             xk      => c_s%xk
             yk      => c_s%yk
             xks     => c_s%xks
             wavexy  => c_s%wavexy
             ug      => c_s%ug
             vg      => c_s%vg
             wls     => c_s%wls
             uls     => c_s%uls
             udrift  => c_s%udrift
             vdrift  => c_s%vdrift
             stokesw => c_s%stokesw
             stokesa => c_s%stokesa
             stokess => c_s%stokess
             stokes  => c_s%stokes
             dtg     => c_s%dtg
             dslg    => c_s%dslg
             dzg     => c_s%dzg
             phip    => c_s%phip
             phiw    => c_s%phiw
             phiv    => c_s%phiv
         return
         end subroutine fill_cs
      end module con_stats