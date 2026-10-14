"""Research implementation: certified tensor Lagrange source gridding + one Gaussian FFT.
Finite-source numerical errors bounded for polynomial interpolation in exact arithmetic;
FFT floating-point error is not included in the analytical certificate.
"""
import math
from dataclasses import dataclass
import numpy as np
from scipy.fft import rfftn, irfftn, next_fast_len
from scipy.special import gammaln
from numba import njit

@njit(cache=True)
def _power_product(points, xmin, dx, sigma, w):
    n=len(points)
    products=np.empty(n,dtype=np.float64)
    for k in range(n):
        q=(points[k]-(xmin+0.5*dx))/dx
        base=int(math.floor(q-0.5*(w-1)+0.5))
        product=1.
        for j in range(w):
            product*=abs((q-base-j)*dx/sigma)
        products[k]=product
    return products

def _err_coeff(w):
    return math.exp(.5*w*math.log(2)+gammaln(.5*(w+1))-.5*math.log(math.pi)-gammaln(w+1))

def interp_bound(points,bounds,nx,ny,sx,sy,wx,wy):
    a=np.asarray(points,dtype=np.float64)
    if len(a)==0:return 0.
    dx=(bounds[1]-bounds[0])/nx; dy=(bounds[3]-bounds[2])/ny
    ex=_err_coeff(wx)*_power_product(a[:,0],bounds[0],dx,sx,wx)
    ey=_err_coeff(wy)*_power_product(a[:,1],bounds[2],dy,sy,wy)
    return float(np.sum(ex+ey+ex*ey))

@njit(cache=True)
def _barycentric_weights(q,w,out):
    start=int(math.floor(q-0.5*(w-1)+0.5))
    exact=-1
    for i in range(w):
        if abs(q-(start+i)) < 2e-12:
            exact=i
            break
    if exact>=0:
        for i in range(w):out[i]=1. if i==exact else 0.
        return start
    # barycentric weights for equidistant nodes: (-1)^i C(w-1,i)/(q-(start+i))
    denom=0.
    binomial=1.
    for i in range(w):
        term=(binomial if i%2==0 else -binomial)/(q-(start+i))
        out[i]=term
        denom+=term
        if i<w-1: binomial=binomial*(w-1-i)/(i+1)
    for i in range(w):out[i]/=denom
    return start

@njit(cache=True)
def _deposit(arr,xmin,ymin,dx,dy,wx,wy,padx,pady,nx,ny):
    grid=np.zeros((nx+2*padx,ny+2*pady),dtype=np.float64)
    a=np.empty(wx,np.float64)
    b=np.empty(wy,np.float64)
    for k in range(len(arr)):
        ix=_barycentric_weights((arr[k,0]-(xmin+0.5*dx))/dx,wx,a)
        iy=_barycentric_weights((arr[k,1]-(ymin+0.5*dy))/dy,wy,b)
        for j in range(wx):
            for h in range(wy):
                grid[ix+j+padx,iy+h+pady] += a[j]*b[h]
    return grid

@dataclass
class InterpolationFFT:
    bounds: tuple
    nx:int
    ny:int
    sigma_x:float
    sigma_y:float
    wx:int
    wy:int
    def __post_init__(self):
        self.dx=(self.bounds[1]-self.bounds[0])/self.nx
        self.dy=(self.bounds[3]-self.bounds[2])/self.ny
        self.padx=max(3,self.wx+2)
        self.pady=max(3,self.wy+2)
        self.mx=self.nx+2*self.padx
        self.my=self.ny+2*self.pady
        self.lx=next_fast_len(self.mx+self.nx-1)
        self.ly=next_fast_len(self.my+self.ny-1)
        # Since output index is shifted by pad, relative offsets lie in [-mx+1, nx+pad-1]
        ux=np.arange(self.lx); ux=np.where(ux<self.nx+self.padx,ux,ux-self.lx)*self.dx/self.sigma_x
        uy=np.arange(self.ly); uy=np.where(uy<self.ny+self.pady,uy,uy-self.ly)*self.dy/self.sigma_y
        gaussian=np.exp(-.5*ux[:,None]**2)*np.exp(-.5*uy[None,:]**2)
        self.khat=rfftn(gaussian)
    def evaluate(self,arr):
        ar=np.asarray(arr,dtype=np.float64)
        if ar.ndim!=2 or ar.shape[1]!=2: raise ValueError('invalid diagram')
        xmin,xmax,ymin,ymax=self.bounds
        if np.any((ar[:,0]<xmin)|(ar[:,0]>xmax)|(ar[:,1]<ymin)|(ar[:,1]>ymax)):
            raise ValueError('points outside bounds')
        mass=_deposit(ar,xmin,ymin,self.dx,self.dy,self.wx,self.wy,self.padx,self.pady,self.nx,self.ny)
        result=irfftn(rfftn(mass,s=(self.lx,self.ly))*self.khat,s=(self.lx,self.ly))
        return result[self.padx:self.padx+self.nx,self.pady:self.pady+self.ny].copy()

def select_width(points,bounds,nx,ny,sx,sy,eps,max_width=16):
    # Compute candidate bounds for each dimension, select least wx*wy satisfying eps.
    a=np.asarray(points,dtype=np.float64)
    dx=(bounds[1]-bounds[0])/nx; dy=(bounds[3]-bounds[2])/ny
    eb=[]
    for pts,x0,step,sig in [(a[:,0],bounds[0],dx,sx),(a[:,1],bounds[2],dy,sy)]:
        eb.append([_err_coeff(w)*_power_product(pts,x0,step,sig,w) for w in range(1,max_width+1)])
    best=None
    for wx in range(1,max_width+1):
        ex=eb[0][wx-1]
        for wy in range(1,max_width+1):
            ey=eb[1][wy-1]
            bound=np.sum(ex+ey+ex*ey)
            if bound<=eps and (best is None or wx*wy<best[0]):
                best=(wx*wy,wx,wy,bound)
    return best

def _even_uniform_1d_error(n,h,sigma,half_width):
    """Sharp uniform interpolation certificate for even 2r-point central stencils.
    Errors controlled in exact arithmetic; fails only from logarithmic arithmetic under/overflow.
    """
    r=half_width
    if r<1:raise ValueError('r>=1 required')
    if n==0:return 0.
    w=2*r
    # uniform product at cell midpoint = ((2r)!/(4^r*r!))**2
    # Gaussian w-derivative constant 1/(2^r*r!).
    logb=(math.log(n)+2*gammaln(2*r+1)-3*gammaln(r+1)
          -5*r*math.log(2)+2*r*math.log(h/sigma))
    return math.exp(logb) if logb < 700 else math.inf

def select_uniform_width(n,bounds,nx,ny,sx,sy,eps,max_width=18):
    dx=(bounds[1]-bounds[0])/nx;dy=(bounds[3]-bounds[2])/ny
    # Per-source certs then multiply n for cross term.
    opts=[]
    for wx in range(2,max_width+1,2):
        ex=_even_uniform_1d_error(1,dx,sx,wx//2)
        for wy in range(2,max_width+1,2):
            ey=_even_uniform_1d_error(1,dy,sy,wy//2)
            e=n*(ex+ey+ex*ey)
            if e<=eps:
                opts.append((wx*wy,wx,wy,e))
    return min(opts) if opts else None
