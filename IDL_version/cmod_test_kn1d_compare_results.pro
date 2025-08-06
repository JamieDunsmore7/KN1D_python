;+
; test_cmod_compare_results.pro
;-


  restore, 'cmod_test_in.sav'

  ; At this point you have in memory:
  ;   x, xlimiter, xsep, GaugeH2, mu, Ti, Te, n, vxi, LC, PipeDia

  Ti = Ti * 1e3 ; convert from keV to eV
  Te = Te * 1e3 ; convert from keV to eV
  n = n * 1e20 ; convert from 1e20 m^-3 to m^-3


  restore, 'test_kn1d_cmod.KN1D_H'

  ;xH,fH,nH,GammaxH,VxH,pH,TH,qxH,qxH_total,NetHSource,Sion,SideWallH,QH,RxH,QH_total,AlbedoH,$

  stop
END
