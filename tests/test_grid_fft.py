import numpy as np
import pytest
from persistence_images._grid_fft import MomentFFT, AdaptiveGridFFT, choose_degree, uniform_error_bound


def direct(points,b, nx,ny,sx,sy):
    xx=b[0]+(np.arange(nx)+.5)*(b[1]-b[0])/nx
    yy=b[2]+(np.arange(ny)+.5)*(b[3]-b[2])/ny
    return np.exp(-.5*((xx[:,None]-points[None,:,0])/sx)**2) @ np.exp(-.5*((yy[None,:]-points[:,None,1])/sy)**2)


@pytest.mark.parametrize('nx,ny,sx,sy,n,eps', [
    (8,9,.4,.3,30,1e-5),
    (64,64,.2,.2,1000,1e-7),
    (48,31,.2,.07,1300,1e-8),
    (128,128,.05,.05,2000,1e-8),
    (96,80,.32,.26,2000,1e-9),
    (1,1,2.,2.,15,1e-9),
    (15,9,.75,.42,150,1e-9),
])
def test_certified_accuracy(nx,ny,sx,sy,n,eps):
    rng=np.random.default_rng(nx*101+ny)
    b=(2.5,3.5,-3.,-2.)
    points=rng.uniform((b[0],b[2]),(b[1],b[3]),size=(n,2))
    app=AdaptiveGridFFT(b,nx,ny,sx,sy,max_degree=26,max_fft_bytes=350_000_000)
    try: image,bound,degree=app.evaluate(points,eps)
    except ValueError:
        pytest.skip('rigorous bound too costly for max-degree on this test')
    reference=direct(points,b,nx,ny,sx,sy)
    assert degree>=0
    assert bound<=eps
    err=np.max(np.abs(reference-image))
    assert err < bound + 2e-10


def test_point_at_edges_and_anisotropic():
    b=(0.,1.,0.,1.)
    points=np.array([[0,0],[1,1],[0,1],[1,0],[.4,.7]])
    a,bound,degree=AdaptiveGridFFT(b,20,32,.2,.3,max_degree=18).evaluate(points,1e-9)
    assert np.max(np.abs(a-direct(points,b,20,32,.2,.3))) <= bound+1e-10


def test_reuse_cache_and_coordinate_order():
    b=(0.,1.,-1.,2.)
    pts=np.array([[.3,.2],[.5,1.5],[.3,.2]])
    engine=AdaptiveGridFFT(b,15,21,.3,.25)
    a,bound,degree=engine.evaluate(pts,1e-8)
    c,_,degree2=engine.evaluate(pts,1e-8)
    assert len(engine._cache)==1 and degree==degree2
    np.testing.assert_array_equal(a,c)
    np.testing.assert_allclose(a.ravel(order='C'),direct(pts,b,15,21,.3,.25).ravel(order='C'),atol=bound+1e-11,rtol=0)


def test_empty_diagram():
    e=MomentFFT((0.,1.,0.,1.),8,9,.2,.2,4)
    np.testing.assert_array_equal(e.evaluate(np.empty((0,2))),np.zeros((8,9)))


def test_out_of_bounds_rejected():
    e=MomentFFT((0.,1.,0.,1.),8,9,.2,.2,4)
    with pytest.raises(ValueError,match='outside'):
        e.evaluate(np.array([[1.01,.2]]))


def test_tolerance_enforcement():
    assert choose_degree(1000,1/128,1/128,.2,.2,1e-8,12) is not None
    assert choose_degree(1000,1/128,1/128,.008,.008,1e-8,6) is None
    with pytest.raises(ValueError,match='certified'):
        AdaptiveGridFFT((0.,1.,0.,1.),128,128,.008,.008,max_degree=6).evaluate(np.array([[.3,.4]]*1000),1e-8)


def test_monotone_error_bounds_in_n():
    v1=uniform_error_bound(100,1/64,1/64,.2,.2,5)
    v2=uniform_error_bound(300,1/64,1/64,.2,.2,5)
    assert v2==pytest.approx(3*v1)


def test_invalid_inputs():
    with pytest.raises(ValueError): MomentFFT((0,0,0,1),8,8,.2,.2,4)
    with pytest.raises(ValueError): MomentFFT((0,1,0,1),8,8,0,.2,4)
    with pytest.raises(ValueError): choose_degree(100,1,1,.2,.2,-1,10)


def test_fft_memory_cap():
    with pytest.raises(MemoryError): MomentFFT((0,1,0,1),128,128,.2,.2,8,max_fft_bytes=1)


def test_actual_source_offsets_improve_certification():
    from persistence_images._grid_fft import tight_degree_for_points
    rng=np.random.default_rng(458)
    pts=rng.random((4000,2))
    b=(0.,1.,0.,1.)
    p,bound=tight_degree_for_points(pts,b,256,256,.2,.2,1e-8,max_degree=12)
    assert p is not None
    assert bound <= uniform_error_bound(len(pts),1/256,1/256,.2,.2,p)
    assert bound<=1e-8


def test_huge_dimensionless_offset_no_overflow():
    assert uniform_error_bound(100,1,1,1e-120,1e-120,4)==float('inf')
