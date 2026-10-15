"""Certified one-FFT approximation of weighted Gaussian *pixel integrals*.

The Gaussian kernel is probability-normalized, with a bounded persistence ramp
weight. Interpolation error bounds concern exact arithmetic; neither signed
deposition nor FFT floating-point errors are included in the certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, isfinite, lgamma, log, pi, sqrt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import Bounds, _points, _validate_params

MAX_STENCIL = 18
MAX_FFT_CELLS = 4_000_000


@dataclass(frozen=True)
class StableFftCertificate:
    width_x: int
    width_y: int
    pixel_error: float
    vector_l1_error: float


def _axis_error(r: int, step: float, sigma: float) -> tuple[float, float]:
    """Source interpolation remainder for pixel mass and full-vector L1."""
    t = step / sigma
    if t <= 0 or not isfinite(t):
        return float("inf"), float("inf")
    common = 2*lgamma(2*r+1)-3*lgamma(r+1)-5*r*log(2)+2*r*log(t)
    pixel = common + log(t/sqrt(2*pi))
    vector = 1.5*lgamma(2*r+1)-2*lgamma(r+1)-4*r*log(2)+2*r*log(t)
    return (exp(pixel) if pixel < 700 else float("inf"),
            exp(vector) if vector < 700 else float("inf"))


def _shape(nx:int, ny:int, wx:int, wy:int) -> tuple[int,int] | None:
    from scipy.fft import next_fast_len
    lx = next_fast_len(2*nx+2*(wx+2)-1, real=True)
    ly = next_fast_len(2*ny+2*(wy+2)-1, real=True)
    return (lx,ly) if lx*ly <= MAX_FFT_CELLS else None


def select_stable_fft_stencil(
    weight_sum: float, bounds: Bounds, nx:int, ny:int,
    sigma_x:float, sigma_y:float, absolute_error:float,
    *, max_width:int=MAX_STENCIL,
) -> StableFftCertificate | None:
    """Find a minimal even stencil certifying per-pixel interpolation error.

    Also reports an independent whole-vector L1 error certificate.
    """
    _validate_params(nx, ny, sigma_x, sigma_y, 1)
    if not isfinite(absolute_error) or absolute_error<=0:
        raise ValueError("absolute_error must be finite and positive")
    if not isfinite(weight_sum) or weight_sum<0:
        raise ValueError("weight_sum must be finite and nonnegative")
    if (not all(isfinite(v) for v in (
        bounds.x_min,bounds.x_max,bounds.y_min,bounds.y_max))
        or bounds.x_max<=bounds.x_min or bounds.y_max<=bounds.y_min):
        raise ValueError("Stable FFT requires finite nondegenerate bounds")
    hx=(bounds.x_max-bounds.x_min)/nx
    hy=(bounds.y_max-bounds.y_min)/ny
    if not (isfinite(hx) and isfinite(hy) and hx>0 and hy>0):
        raise ValueError("Invalid pixel width")
    if max_width<2 or max_width>MAX_STENCIL:
        raise ValueError("max_width must be between 2 and 18")
    if weight_sum==0:
        return StableFftCertificate(2,2,0.,0.)
    best=None
    for wx in range(2,max_width+1,2):
        ex,bx=_axis_error(wx//2,hx,sigma_x)
        for wy in range(2,max_width+1,2):
            ey,by=_axis_error(wy//2,hy,sigma_y)
            e=weight_sum*(ex+ey+ex*ey)
            if e<=absolute_error and _shape(nx,ny,wx,wy) is not None:
                cert=StableFftCertificate(wx,wy,e,weight_sum*(bx+by+bx*by))
                candidate=(wx*wy,max(wx,wy),wx,wy,cert)
                if best is None or candidate[:4]<best[:4]:
                    best=candidate
    return None if best is None else best[-1]


try:
    from numba import njit
except ImportError:
    njit=None

if njit is not None:
    @njit(cache=True)
    def _barycentric_weights(q,w,out):
        start=int(np.floor(q-.5*(w-1)+.5))
        exact=-1
        for i in range(w):
            if q == start+i:
                exact=i
        if exact>=0:
            for i in range(w):
                out[i]=1. if i==exact else 0.
            return start
        total=0.
        binomial=1.
        for i in range(w):
            out[i]=(binomial if i%2==0 else -binomial)/(q-(start+i))
            total+=out[i]
            if i+1<w:
                binomial=binomial*(w-1-i)/(i+1)
        for i in range(w):
            out[i]/=total
        return start

    @njit(cache=True)
    def _deposit_weighted(points,scale,wx,wy,padx,pady,mass,xmin,ymin,hx,hy):
        ax=np.empty(wx,dtype=np.float64)
        ay=np.empty(wy,dtype=np.float64)
        for k in range(points.shape[0]):
            b,p=points[k,0],points[k,1]
            w=min(1.,p/scale)
            if w==0.:
                continue
            ix=_barycentric_weights((b-(xmin+.5*hx))/hx,wx,ax)
            iy=_barycentric_weights((p-(ymin+.5*hy))/hy,wy,ay)
            for i in range(wx):
                for j in range(wy):
                    mass[ix+i+padx,iy+j+pady]+=w*ax[i]*ay[j]


@lru_cache(maxsize=8)
def _cached_integrated_spectrum(nx:int,ny:int,hx:float,hy:float,
        sx:float,sy:float,wx:int,wy:int):
    from scipy.fft import rfft2
    from scipy.special import ndtr
    shape=_shape(nx,ny,wx,wy)
    if shape is None:
        raise MemoryError("Integrated FFT workspace exceeds memory cap")
    lx,ly=shape
    def axis_mass(n,h,sigma):
        distance=np.minimum(np.arange(n),n-np.arange(n)).astype(np.float64)
        lo=(distance-.5)*h/sigma
        hi=(distance+.5)*h/sigma
        return ndtr(-lo)-ndtr(-hi)
    spectrum=rfft2(axis_mass(lx,hx,sx)[:,None] *
                   axis_mass(ly,hy,sy)[None,:])
    spectrum.flags.writeable=False
    return spectrum,shape


class StableIntegratedFft:
    """Reusable cached integrated-Gaussian FFT kernel and interpolation stencil."""
    def __init__(self,bounds:Bounds,nx:int,ny:int,sigma_x:float,sigma_y:float,
                 certificate:StableFftCertificate,weight_scale:float):
        _validate_params(nx,ny,sigma_x,sigma_y,1)
        if not isfinite(weight_scale) or weight_scale<=0:
            raise ValueError("weight_scale must be positive and finite")
        self.bounds,self.nx,self.ny=bounds,nx,ny
        self.weight_scale=weight_scale
        self.wx,self.wy=certificate.width_x,certificate.width_y
        self.px,self.py=self.wx+2,self.wy+2
        self.hx=(bounds.x_max-bounds.x_min)/nx
        self.hy=(bounds.y_max-bounds.y_min)/ny
        self.spectrum,self.shape=_cached_integrated_spectrum(
            nx,ny,self.hx,self.hy,sigma_x,sigma_y,self.wx,self.wy)
        self.certificate=certificate

    def evaluate(self,diagram:ArrayLike)->NDArray[np.float64]:
        if njit is None:
            raise ImportError("Stable integrated FFT requires Numba and SciPy; install .[grid]")
        from scipy.fft import rfft2,irfft2
        points=_points(diagram)
        if np.any(points[:,1]<0):
            raise ValueError("Persistence must be nonnegative")
        b=self.bounds
        if (np.any(points[:,0]<b.x_min) or np.any(points[:,0]>b.x_max)
            or np.any(points[:,1]<b.y_min) or np.any(points[:,1]>b.y_max)):
            raise ValueError("Sources outside fixed grid; use direct CDF integration")
        if not len(points) or not np.any(points[:,1]):
            return np.zeros(self.nx*self.ny,dtype=np.float64)
        mass=np.zeros(self.shape,dtype=np.float64)
        _deposit_weighted(points,self.weight_scale,self.wx,self.wy,self.px,self.py,
            mass,b.x_min,b.y_min,self.hx,self.hy)
        convolved=irfft2(rfft2(mass)*self.spectrum,s=self.shape)
        result=convolved[self.px:self.px+self.nx,self.py:self.py+self.ny].copy()
        # Projection onto nonnegative values cannot increase error against
        # the mathematically nonnegative exact pixel integrals.
        np.maximum(result,0.,out=result)
        return result.ravel(order="C")


def stable_interp_fft_image(diagram:ArrayLike,bounds:Bounds,nx:int=10,ny:int=8,
        sigma_x:float=.86,sigma_y:float=.63984,*,weight_scale:float=1.,
        absolute_error:float=1e-8,return_certificate:bool=False):
    """Certified integrated one-FFT; raise if tolerance cannot be certified."""
    points=_points(diagram)
    if np.any(points[:,1]<0):
        raise ValueError("Persistence must be nonnegative")
    if not isfinite(weight_scale) or weight_scale<=0:
        raise ValueError("weight_scale must be finite and positive")
    weight_sum=float(np.minimum(points[:,1]/weight_scale,1.).sum())
    cert=select_stable_fft_stencil(weight_sum,bounds,nx,ny,
        sigma_x,sigma_y,absolute_error)
    if cert is None:
        raise ValueError("No integrated FFT stencil certifies absolute_error")
    if weight_sum==0:
        image=np.zeros(nx*ny,dtype=np.float64)
    else:
        image=StableIntegratedFft(bounds,nx,ny,sigma_x,sigma_y,cert,weight_scale).evaluate(points)
    return (image,cert) if return_certificate else image
