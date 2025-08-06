;+
; test_kn1d_cmod_from_sav.pro
;-


  ; 1) Restore all the inputs you previously SAVE’d
  restore, 'cmod_test_in.sav'

  ; At this point you have in memory:
  ;   x, xlimiter, xsep, GaugeH2, mu, Ti, Te, n, vxi, LC, PipeDia

  Ti = Ti * 1e3 ; convert from keV to eV
  Te = Te * 1e3 ; convert from keV to eV
  n = n * 1e20 ; convert from 1e20 m^-3 to m^-3

  ; 2) (Optional) set any keyword defaults you want to mirror the Python run

  refine         = 0     ; start fresh
  File           = 'test_kn1d_cmod'
  ReadInput      = 0     ; don’t re‐load a .KN1D_input here
  NewFile        = 1     ; allow writing out mesh+data at end
  compute_errors = 1
  Hdebrief = 1
  H2debrief = 1

  ;KN1D, x, xlimiter, xsep, GaugeH2, mu, Ti, Te, n, vxi, LC, PipeDia





  ; 3) Call the core KN1D routine with exactly the same signature
  KN1D, x, xlimiter, xsep, GaugeH2, mu, Ti, Te, n, vxi, LC, PipeDia, $
        truncate=truncate, refine=refine, File=File, ReadInput=ReadInput, NewFile=NewFile, $
        debrief=debrief, debug=debug, compute_errors=compute_errors, Hdebrief=Hdebrief, H2debrief=H2debrief

  stop

end
