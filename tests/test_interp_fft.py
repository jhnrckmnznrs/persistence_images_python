import numpy as np
import pytest
from persistence_images import Bounds,persistence_image
from persistence_images._interp_fft import (
    InterpolationFFT,interp_bound,select_uniform_width,select_width,
    _even_uniform_1d_error,
)

@pytest.mark.parametrize('shape,bandwidth', [((16,20),(.3,.2)),((32,24),(.16,.4)),((64,48),(.02,.3)),((64,64),(.3,.3))])
def test_certified_various_shapes(shape,bandwidth):
    nx,ny=shape; sx,sy=bandwidth
    rng=np.random.default_rng(nx+ny)
    arr=rng.uniform(size=(175,2))
    b=(0.,1.,0.,1.); tol=1.e-6
    selected=select_uniform_width(len(arr),b,nx,ny,sx,sy,tol,max_width=22)
    if selected is None:
        pytest.skip('bounds deliberately too conservative for this sigma/grid')
    cost,wx,wy,bound=selected
    out=InterpolationFFT(b,nx,ny,sx,sy,wx,wy).evaluate(arr)
    exact=persistence_image(arr,Bounds(*b),nx,ny,sx,sy,engine='dense').reshape(nx,ny)
    assert np.max(np.abs(out-exact)) <= max(tol,1.e-9) # roundoff excluded from certificate
    assert bound<=tol
    assert interp_bound(arr,b,nx,ny,sx,sy,wx,wy) <= bound*(1+1e-8)

@pytest.mark.parametrize('center', [0.0,0.1,0.5,0.99,1.0])
def test_edge_sources_and_ghost_nodes(center):
    arr=np.array([[center,center],[center,0.5],[0.75,center]])
    b=(0.,1.,0.,1.); nx=32;ny=24
    selected=select_uniform_width(len(arr),b,nx,ny,.4,.2,1e-8,max_width=18)
    assert selected is not None
    _,wx,wy,bound=selected
    out=InterpolationFFT(b,nx,ny,.4,.2,wx,wy).evaluate(arr)
    exact=persistence_image(arr,Bounds(*b),nx,ny,.4,.2,engine='dense').reshape(nx,ny)
    assert np.max(np.abs(out-exact)) <= 1e-8


def test_axis_anisotropy_nonzero_origin():
    b=(-2.,3.,4.,8.); n=70
    rng=np.random.default_rng(77)
    pts=np.column_stack((rng.uniform(b[0],b[1],n),rng.uniform(b[2],b[3],n)))
    selected=select_uniform_width(n,b,144,96,.9,.5,1e-7,max_width=20)
    assert selected is not None
    _,wx,wy,err=selected
    out=InterpolationFFT(b,144,96,.9,.5,wx,wy).evaluate(pts)
    dense=persistence_image(pts,Bounds(*b),144,96,.9,.5,engine='dense').reshape(144,96)
    assert np.max(np.abs(out-dense))<1e-7


def test_integrated_engine_and_flatten_order():
    rng=np.random.default_rng(4)
    pts=rng.uniform(size=(300,2))
    b=Bounds(0,1,0,1)
    out=persistence_image(pts,b,64,40,.35,.25,engine='interp_fft',absolute_error=1e-8)
    exact=persistence_image(pts,b,64,40,.35,.25,engine='dense')
    assert out.shape==(64*40,)
    assert np.max(np.abs(out-exact))<1e-8


def test_error_requires_explicit_tolerance():
    with pytest.raises(ValueError):
        persistence_image(np.array([[0.5,0.5]]),Bounds(0,1,0,1),
                          32,32,.2,.2,engine='interp_fft')


def test_adaptive_rejects_unresolved_gaussian():
    with pytest.raises(ValueError):
        persistence_image(np.array([[0.5,0.5]]*30),Bounds(0,1,0,1),
                          12,12,.0001,.0001,engine='interp_fft',absolute_error=1e-10)


def test_repeated_points():
    pts=np.repeat(np.array([[.22,.67],[.81,.15]]), 100, axis=0)
    b=(0,1,0,1); nx=40; ny=40
    selected=select_uniform_width(len(pts),b,nx,ny,.25,.3,1e-8)
    assert selected is not None
    _,wx,wy,_=selected
    approx=InterpolationFFT(b,nx,ny,.25,.3,wx,wy).evaluate(pts)
    dense=persistence_image(pts,Bounds(*b),nx,ny,.25,.3,engine='dense').reshape(nx,ny)
    assert np.max(np.abs(approx-dense))<1e-8


def test_uniform_product_bound_at_cell_midpoint():
    for halfwidth in range(1,7):
        w=2*halfwidth
        ds=np.arange(-halfwidth+1,halfwidth+1,dtype=float)
        v=np.prod(np.abs(.5-ds))
        from scipy.special import gammaln
        certified=np.exp(2*gammaln(2*halfwidth+1)-2*gammaln(halfwidth+1)-4*halfwidth*np.log(2))
        assert v==pytest.approx(certified, rel=1e-12)


def test_empty_image():
    b=Bounds(0,1,0,1)
    image=persistence_image(np.empty((0,2)),b,16,18,.2,.2,engine='interp_fft',absolute_error=1e-8)
    assert np.all(image==0.)
