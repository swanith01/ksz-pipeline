! external/amber_patch/healpix_stub.f90
!
! [ksz-pipeline] Stand-in for the four HEALPix/FITS modules AMBER's cmb.f90
! and cmbreion.f90 `use` (fitstools, head_fits, pix_tools, alm_tools), so
! AMBER compiles and links WITHOUT HEALPix/cfitsio installed.
!
! These routines are only reached when the CMB "Map" input is make/write
! (full-sky HEALPix map-making). The ksz-pipeline adapter never needs that:
! it reads power_*.txt, cl_ksz.txt and the fields_*.dat dump. Keep Map = no.
! If anything does call a stub, the run STOPS with a clear message rather
! than producing silent garbage.
!
! Signatures mirror exactly how AMBER calls them (checked against upstream
! main, 2026-09-10). Replace with real HEALPix later if maps are wanted:
! build HEALPix + cfitsio and restore the original Makefile link lines.

module head_fits
  implicit none
contains
  subroutine write_minimal_header(header, dtype, nside, ordering, coordsys, units)
    character(len=*), dimension(:), intent(inout) :: header
    character(len=*), intent(in) :: dtype
    integer(4), intent(in), optional :: nside
    character(len=*), intent(in), optional :: ordering, coordsys, units
    stop 'healpix_stub: write_minimal_header called -- set CMB Map = no'
  end subroutine write_minimal_header
end module head_fits


module fitstools
  implicit none
contains
  subroutine write_bintab(map, npix, nmap, header, nlheader, filename)
    real(4), dimension(0:,1:), intent(in) :: map
    integer(8), intent(in) :: npix
    integer(4), intent(in) :: nmap, nlheader
    character(len=*), dimension(:), intent(in) :: header
    character(len=*), intent(in) :: filename
    stop 'healpix_stub: write_bintab called -- set CMB Map = no'
  end subroutine write_bintab
end module fitstools


module pix_tools
  implicit none
contains
  subroutine vec2ang(vector, theta, phi)
    real(8), dimension(1:), intent(in) :: vector
    real(8), intent(out) :: theta, phi
    theta = 0; phi = 0
    stop 'healpix_stub: vec2ang called -- set CMB Map = no'
  end subroutine vec2ang

  subroutine ang2pix_ring(nside, theta, phi, ipix)
    integer(4), intent(in) :: nside
    real(8), intent(in) :: theta, phi
    integer(8), intent(out) :: ipix
    ipix = 0
    stop 'healpix_stub: ang2pix_ring called -- set CMB Map = no'
  end subroutine ang2pix_ring
end module pix_tools


module alm_tools
  implicit none
contains
  subroutine map2alm(nsmax, nlmax, nmmax, map, alm, zbounds, w8ring)
    integer(4), intent(in) :: nsmax, nlmax, nmmax
    real(8), dimension(0:), intent(in) :: map
    complex(8), dimension(1:,0:,0:), intent(out) :: alm
    real(8), dimension(1:2), intent(in) :: zbounds
    real(8), dimension(1:,1:), intent(in) :: w8ring
    alm = 0
    stop 'healpix_stub: map2alm called -- set CMB Map = no'
  end subroutine map2alm

  subroutine alm2cl(nlmax, nmmax, alm, cl)
    integer(4), intent(in) :: nlmax, nmmax
    complex(8), dimension(1:,0:,0:), intent(in) :: alm
    real(8), dimension(0:,1:), intent(out) :: cl
    cl = 0
    stop 'healpix_stub: alm2cl called -- set CMB Map = no'
  end subroutine alm2cl
end module alm_tools
