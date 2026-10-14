"""Grid-aware Gaussian moment/FFT backend for point-sampled persistence images.

Independent research code, MIT compatible. Input diagram is (birth, persistence).
This uses Gaussian-Taylor moments on the *source lattice*, and zero-padded
FFT convolution to evaluate at all image centers, with a rigorous bound on
Taylor truncation in exact arithmetic. No claim of total FP error bounds.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import factorial, gamma, log, exp, sqrt, pi, hypot, isfinite
import numpy as np
from scipy.fft import rfftn, irfftn, next_fast_len
from numba import njit


def moment_indices(degree: int):
    return [(a, total - a) for total in range(degree + 1) for a in range(total + 1)]


def uniform_error_bound(n: int, dx: float, dy: float, sx: float, sy: float, degree: int) -> float:
    """N * E|Z|^(p+1) * max_source_offset^(p+1)/(p+1)!

    Characteristic-function proof controls the real, unnormalized Gaussian.
    Assumes each source is in grid domain and clipped to nearest pixel center.
    """
    if n < 0 or not all(isfinite(v) and v > 0 for v in (dx,dy,sx,sy)):
        raise ValueError("invalid data/bandwidth")
    if not n: return 0.0
    r = hypot(dx/(2*sx), dy/(2*sy))
    if not r: return 0.0
    from scipy.special import gammaln
    m = degree + 1
    ln = log(n) + (m/2)*log(2) + gammaln((m+1)/2) - .5*log(pi) - gammaln(m+1) + m*log(r)
    return float("inf") if ln > 709.0 else float(exp(ln))


def choose_degree(n, dx, dy, sx, sy, eps, max_degree=16):
    if not np.isfinite(eps) or eps <= 0: raise ValueError("eps must be positive and finite")
    for p in range(max_degree + 1):
        if uniform_error_bound(n,dx,dy,sx,sy,p) <= eps:
            return p
    return None


@njit(cache=True)
def _offset_power_sums(points, x_min, y_min, dx, dy, sx, sy, nx, ny, max_degree):
    # The same assignment and displacement formula as _deposit.
    # Returns sum_k ||(source_k-center_k)/sigma||^(m), m=1..max_degree+1.
    sums=np.zeros(max_degree+1,dtype=np.float64)
    for k in range(points.shape[0]):
        ix=int(np.floor((points[k,0]-x_min)/dx))
        iy=int(np.floor((points[k,1]-y_min)/dy))
        ix=max(0,min(ix,nx-1))
        iy=max(0,min(iy,ny-1))
        ux=(points[k,0]-(x_min+(ix+0.5)*dx))/sx
        uy=(points[k,1]-(y_min+(iy+0.5)*dy))/sy
        radius=np.sqrt(ux*ux+uy*uy)
        power=radius
        for m in range(max_degree+1):
            sums[m] += power
            power *= radius
    return sums


def tight_degree_for_points(points, bounds, nx, ny, sx, sy, eps, max_degree=12):
    """Tighter point-dependent uniform error certificate.

    For m=p+1 and standardized source offsets h_k from their nearest
    lattice pixel centers, E|Z|^m/m! * SUM_k ||h_k||^m bounds the total
    Taylor truncation uniformly over all target pixels, in exact arithmetic.
    This improves over N*max(||h_k||)^m without relying on cancellation.
    """
    from scipy.special import gammaln
    arr=np.asarray(points,dtype=np.float64)
    if arr.size==0:
        return 0,0.0
    if arr.ndim!=2 or arr.shape[1]!=2 or not np.all(np.isfinite(arr)):
        raise ValueError('expected finite (N,2) diagram')
    xmin,xmax,ymin,ymax=bounds
    if not (np.all(arr[:,0]>=xmin) and np.all(arr[:,0]<=xmax) and
            np.all(arr[:,1]>=ymin) and np.all(arr[:,1]<=ymax)):
        raise ValueError('diagram outside bounds')
    sums=_offset_power_sums(arr,xmin,ymin,(xmax-xmin)/nx,(ymax-ymin)/ny,
                            sx,sy,nx,ny,max_degree)
    for p,s in enumerate(sums):
        m=p+1
        constant=exp((m/2)*log(2)+gammaln((m+1)/2)-.5*log(pi)-gammaln(m+1))
        bound=constant*s
        if bound<=eps:
            return p,float(bound)
    return None,None


@njit(cache=True)
def _deposit(points, x_min, y_min, dx, dy, sx, sy, nx, ny, aidx, bidx, degree):
    # One pass per point, O(N p^2), no additional data structures.
    moments = np.zeros((aidx.size, nx, ny), dtype=np.float64)
    px=np.empty(degree+1, dtype=np.float64)
    py=np.empty(degree+1, dtype=np.float64)
    for k in range(points.shape[0]):
        ix=int(np.floor((points[k,0]-x_min)/dx))
        iy=int(np.floor((points[k,1]-y_min)/dy))
        if ix<0:ix=0
        if iy<0:iy=0
        if ix>=nx: ix=nx-1
        if iy>=ny: iy=ny-1
        ux=(points[k,0]-(x_min+(ix+0.5)*dx))/sx
        uy=(points[k,1]-(y_min+(iy+0.5)*dy))/sy
        px[0]=1.
        py[0]=1.
        for h in range(1,degree+1):
            px[h]=px[h-1]*ux/h
            py[h]=py[h-1]*uy/h
        for q in range(aidx.size):
            moments[q,ix,iy] += px[aidx[q]] * py[bidx[q]]
    return moments


def _hermite_probnorm(z, deg):
    vals=np.empty((deg+1,len(z)),dtype=np.float64)
    vals[0]=1
    if deg:
        vals[1]=z
    for m in range(1,deg):
        vals[m+1]=z*vals[m]-m*vals[m-1]
    return vals


@dataclass
class MomentFFT:
    bounds: tuple[float,float,float,float] # xmin xmax ymin ymax
    nx: int
    ny: int
    sigma_x: float
    sigma_y: float
    degree: int
    max_fft_bytes: int = 512_000_000

    def __post_init__(self):
        if not isinstance(self.nx, (int, np.integer)) or not isinstance(self.ny,(int,np.integer)):
            raise ValueError("grid dimensions must be integers")
        if not isinstance(self.degree, (int,np.integer)) or self.degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        self.nx=int(self.nx); self.ny=int(self.ny)
        if min(self.nx,self.ny) < 1 or not all(isfinite(s) and s > 0 for s in (self.sigma_x,self.sigma_y)):raise ValueError('invalid grid/bandwidth')
        xmin,xmax,ymin,ymax=map(float,self.bounds)
        if not all(isfinite(t) for t in (xmin,xmax,ymin,ymax)) or xmax<=xmin or ymax<=ymin:
            raise ValueError('grid bounds must be nondegenerate')
        self.dx=(xmax-xmin)/self.nx
        self.dy=(ymax-ymin)/self.ny
        self.indices=moment_indices(self.degree)
        self.aidx=np.array([i for i,j in self.indices], dtype=np.int64)
        self.bidx=np.array([j for i,j in self.indices], dtype=np.int64)
        self.lx=next_fast_len(2*self.nx-1)
        self.ly=next_fast_len(2*self.ny-1)
        self.estimated_fft_bytes = len(self.indices)*self.lx*(self.ly//2+1)*16
        # Bound kernel spectra + per-image moment spectra + source moments,
        # plus a reasonable allowance for FFT workspace.
        self.estimated_working_bytes = (3*self.estimated_fft_bytes
                + len(self.indices)*self.nx*self.ny*8)
        if self.estimated_working_bytes > self.max_fft_bytes:
            raise MemoryError('Gaussian moment FFT working arrays exceed memory limit')
        # Periodic-array differences. Padding prevents convolution aliasing
        # for source and target sites in [0,nx) x [0,ny).
        ii=np.arange(self.lx); ii=np.where(ii < self.nx, ii, ii-self.lx)
        jj=np.arange(self.ly); jj=np.where(jj < self.ny, jj, jj-self.ly)
        ux=ii*self.dx/self.sigma_x
        uy=jj*self.dy/self.sigma_y
        hx=_hermite_probnorm(ux,self.degree)
        hy=_hermite_probnorm(uy,self.degree)
        gx=np.exp(-.5*ux**2)
        gy=np.exp(-.5*uy**2)
        # Clip invalid offset positions in the wrap gap (they never influence output),
        # the arrays may contain nonzero values there without any effect.
        self.kernel_fft=np.empty((len(self.indices),self.lx,self.ly//2+1),dtype=np.complex128)
        for m,(a,b) in enumerate(self.indices):
            kernel=(hx[a]*gx)[:,None]*(hy[b]*gy)[None,:]
            self.kernel_fft[m]=rfftn(kernel)

    def evaluate(self, points):
        arr=np.asarray(points,dtype=np.float64)
        if arr.size==0:
            return np.zeros((self.nx,self.ny),dtype=np.float64)
        if arr.ndim!=2 or arr.shape[1]!=2 or not np.all(np.isfinite(arr)):
            raise ValueError('expected finite (N,2) birth-persistence diagram')
        x0,x1,y0,y1=self.bounds
        if np.any(arr[:,0]<x0) or np.any(arr[:,0]>x1) or np.any(arr[:,1]<y0) or np.any(arr[:,1]>y1):
            raise ValueError('diagram points outside image bounds; error certification would fail')
        moments=_deposit(arr,x0,y0,self.dx,self.dy,self.sigma_x,self.sigma_y,self.nx,self.ny,self.aidx,self.bidx,self.degree)
        # Shared FFT precomputed for the entire batch; only moments change.
        mom_fft=rfftn(moments,s=(self.lx,self.ly),axes=(-2,-1))
        weighted=np.einsum('ijk,ijk->jk',mom_fft,self.kernel_fft,optimize=False)
        img=irfftn(weighted,s=(self.lx,self.ly))
        return img[:self.nx,:self.ny].copy()

    def truncation_bound(self,n):
        return uniform_error_bound(n,self.dx,self.dy,self.sigma_x,self.sigma_y,self.degree)


class AdaptiveGridFFT:
    """Select certified moment degree, or decline with a clear reason."""
    def __init__(self,bounds,nx,ny,sigma_x,sigma_y,max_degree=12,max_fft_bytes=512_000_000):
        self.params=(bounds,nx,ny,sigma_x,sigma_y)
        self.max_degree=max_degree
        self.max_fft_bytes=max_fft_bytes
        self._cache={}

    def evaluate(self,points,eps=1e-8):
        _,nx,ny,sx,sy=self.params
        xmin,xmax,ymin,ymax=self.params[0]
        degree,bound=tight_degree_for_points(points,self.params[0],nx,ny,sx,sy,eps,self.max_degree)
        if degree is None:
            raise ValueError('Requested tolerance cannot be certified at maximum moment degree; use dense/sparse instead')
        if degree not in self._cache:
            self._cache[degree]=MomentFFT(*self.params,degree,max_fft_bytes=self.max_fft_bytes)
        return self._cache[degree].evaluate(points),bound,degree
