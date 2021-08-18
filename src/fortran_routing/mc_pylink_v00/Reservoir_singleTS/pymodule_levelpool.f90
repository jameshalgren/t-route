module levelpool_interface

use, intrinsic :: iso_c_binding, only: c_double
use module_levelpool, only: levelpool_physics

implicit none
contains
subroutine c_levelpool_physics(dt,qi0,qi1,ql,&
    ar,we,maxh,wc,wl,dl,oe,oc,oa,H0,H1,qo1) bind(c)

    real(c_double), intent(in) :: dt
    real(c_double), intent(in) :: qi0, qi1, ql
    real(c_double), intent(in) :: ar,we,maxh,wc,wl,dl,oe,oc,oa
    real(c_double), intent(in) :: H0
    real(c_double), intent(out) :: H1,qo1

    call levelpool_physics(dt,qi0,qi1,ql,&
        ar,we,maxh,wc,wl,dl,oe,oc,oa,H0,H1,qo1) 

end subroutine c_levelpool_physics
end module levelpool_interface

