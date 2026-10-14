import numpy as np
import pytest
from persistence_images import Bounds, persistence_image, batch_persistence_images


def test_explicit_grid_engine_and_batch():
    rng=np.random.default_rng(101)
    b=Bounds(0,1,0,1)
    pts=rng.uniform(0,1,(1500,2))
    img=persistence_image(pts,b,128,128,.2,.15,engine='grid_fft',absolute_error=1e-8)
    ref=persistence_image(pts,b,128,128,.2,.15,engine='dense')
    np.testing.assert_allclose(img,ref,atol=1e-8,rtol=0)
    batch=batch_persistence_images([pts,pts[:300]],b,nx=128,ny=128,sigma_x=.2,sigma_y=.15,
                                   engine='grid_fft',absolute_error=1e-8)
    np.testing.assert_allclose(batch[0],img,atol=1e-9,rtol=0)
    np.testing.assert_allclose(batch[1],persistence_image(pts[:300],b,128,128,.2,.15,
                                engine='dense'),atol=1e-8,rtol=0)


def test_grid_requires_explicit_error():
    with pytest.raises(ValueError,match='absolute_error'):
        persistence_image(np.array([[.2,.3]]), Bounds(0,1,0,1),128,128,.2,.2,engine='grid_fft')


def test_auto_large_diagram_chooses_grid_and_agrees():
    rng=np.random.default_rng(12)
    pts=rng.random((42000,2));b=Bounds(0,1,0,1)
    img=persistence_image(pts,b,128,128,.2,.2,engine='auto',absolute_error=1e-8)
    ref=persistence_image(pts,b,128,128,.2,.2,engine='dense')
    np.testing.assert_allclose(img,ref,atol=1e-8,rtol=0)
