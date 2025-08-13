;+------------------------------------------------------------------
; test_kn1d_d3d_from_h5.pro
; Read inputs from an HDF5 file and run KN1D with the same signature
; as your Python call.
;+------------------------------------------------------------------
pro test_kn1d_d3d_from_h5

  file = 'D3D_inputs.h5'
  fid  = H5F_OPEN(file, /READ)

  ; Helper for tidy open/read/close
  forward_function _h5read
  function _h5read, fid, name
    did = H5D_OPEN(fid, name)
    val = H5D_READ(did)
    H5D_CLOSE, did
    return, val
  end

  ; --- Read datasets written by the Python script ---
  d_pipe  = _h5read(fid, 'D_pipe')   ; PipeDia
  lc      = _h5read(fid, 'L_c')      ; LC
  mu      = _h5read(fid, 'mu')
  n       = _h5read(fid, 'n')        ; density
  p_wall  = _h5read(fid, 'H2Gauge')  ; GaugeH2
  t_e     = _h5read(fid, 'Te')
  t_i     = _h5read(fid, 'Ti')
  vx      = _h5read(fid, 'vx')       ; vxi
  x       = _h5read(fid, 'x')
  x_lim   = _h5read(fid, 'x_lim')
  x_sep   = _h5read(fid, 'x_sep')

  H5F_CLOSE, fid

  ; If your PKL already contained values in the final units
  ; (e.g., Te/Ti in eV, n in m^-3), no further conversion needed.
  ; Uncomment if you do need unit adjustments:
  ; t_e = t_e * 1d3   ; keV -> eV
  ; t_i = t_i * 1d3   ; keV -> eV
  ; n   = n  * 1d20   ; 1e20 m^-3 -> m^-3

  ; --- Mirror your Python keyword choices ---
  refine         = 0L
  File           = 'test_kn1d_d3d'
  ReadInput      = 0
  NewFile        = 1
  compute_errors = 1
  Hdebrief       = 1
  H2debrief      = 1

  ; --- Call KN1D with the same positional args as Python ---
  KN1D, x, x_lim, x_sep, p_wall, mu, t_i, t_e, n, vx, lc, d_pipe, $
        refine=refine, File=File, ReadInput=ReadInput, NewFile=NewFile, $
        compute_errors=compute_errors, Hdebrief=Hdebrief, H2debrief=H2debrief

  stop
end
