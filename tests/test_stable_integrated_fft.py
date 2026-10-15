"""Regression tests for stable weighted Gaussian pixel-integral FFT."""
import numpy as np
import pytest
pytest.importorskip("scipy")
pytest.importorskip("numba")
from persistence_images import (
    Bounds, stable_persistence_image, stable_batch_images,
    stable_interp_fft_image, select_stable_fft_stencil,
)

@pytest.mark.parametrize("nx,ny,sx,sy,b",[
    (64,48,.2,.3,Bounds(0,1,0,1)),
    (96,80,.4,.23,Bounds(-2,3,0,1)),
    (80,64,.1,.26,Bounds(0,1,0,1)),
])
def test_certificates(nx,ny,sx,sy,b):
    rng=np.random.default_rng(87)
    pts=np.column_stack([rng.uniform(b.x_min,b.x_max,220),
                         rng.uniform(b.y_min,b.y_max,220)])
    fft,cert=stable_interp_fft_image(pts,b,nx,ny,sx,sy,
        weight_scale=.6,absolute_error=1e-8,return_certificate=True)
    exact=stable_persistence_image(pts,b,nx,ny,sx,sy,weight_scale=.6,engine="dense")
    err=np.abs(fft-exact)
    assert cert.pixel_error<=1e-8
    assert err.max()<=cert.pixel_error+2e-10
    assert err.sum()<=cert.vector_l1_error+2e-8
    assert np.all(fft>=0.)

def test_boundary_and_diagonal():
    b=Bounds(0,1,0,1)
    pts=np.array([[0.,0.],[1.,1.],[.2,.1],[.5,0.],[0.,1.]])
    v=stable_interp_fft_image(pts,b,64,64,.2,.2,weight_scale=.5,absolute_error=1e-8)
    ref=stable_persistence_image(pts,b,64,64,.2,.2,weight_scale=.5,engine="dense")
    assert np.max(abs(v-ref)) <= 1.02e-8
    assert not np.any(stable_interp_fft_image([[.5,0.]],b,64,64,.2,.2,absolute_error=1e-8))

def test_outside_and_fallback():
    b=Bounds(0,1,0,1)
    pts=np.array([[.5,1.4],[.3,.1]])
    with pytest.raises(ValueError,match="outside"):
        stable_interp_fft_image(pts,b,64,64,.2,.2,absolute_error=1e-8)
    np.testing.assert_allclose(
        stable_persistence_image(pts,b,64,64,.2,.2,engine="auto",absolute_error=1e-8),
        stable_persistence_image(pts,b,64,64,.2,.2,engine="dense"),atol=1e-12)

def test_narrow_kernel_rejected():
    b=Bounds(0,1,0,1)
    assert select_stable_fft_stencil(15.,b,16,16,1e-4,1e-4,1e-8) is None
    with pytest.raises(ValueError,match="certif"):
        stable_interp_fft_image([[.4,.5]]*30,b,16,16,1e-4,1e-4,absolute_error=1e-8)

def test_mixed_spacing_batch():
    rng=np.random.default_rng(17)
    diagrams=[rng.uniform(.1,.6,(600,2)),rng.uniform(.1,.6,(900,2))]
    b=Bounds(0,1.5,0,1.5)
    sizes=[1.,1.4]
    fast=stable_batch_images(diagrams,b,96,80,.25,.19,spacings=sizes,
        weight_scale=.6,engine="interp_fft",absolute_error=1e-8)
    direct=stable_batch_images(diagrams,b,96,80,.25,.19,spacings=sizes,
        weight_scale=.6,engine="dense")
    for x,y in zip(fast,direct):
        assert np.max(np.abs(x-y))<=1.01e-8

def test_bad_options():
    b=Bounds(0,1,0,1)
    with pytest.raises(ValueError):
        stable_persistence_image([(0.4,.5)],b,engine="interp_fft")
    with pytest.raises(ValueError):
        stable_interp_fft_image([(0.4,-.5)],b,absolute_error=1e-8)
    with pytest.raises(ValueError):
        stable_interp_fft_image([(0.4,.5)],b,absolute_error=0.)
    with pytest.raises(ValueError):
        stable_interp_fft_image([(0.4,.5)],Bounds(0,0,0,1),absolute_error=1e-8)
