"""Exact CDF-basis tests for stable, weighted pixel-integral images."""
import numpy as np
import pytest
from persistence_images import Bounds, StableCdfBasis, stable_batch_images, stable_persistence_image

B=Bounds(-2.0, 3.0, -1.0, 4.0)

def test_cdf_basis_anisotropic_and_diagonal():
    a=np.array([(0.2,0.1),(0.2,0.1),(0.8,1.2),(0.8,1.2),(1.1,0.0)])
    b=np.array([(0.2,1.2),(0.8,0.1),(0.8,0.0)])
    basis=StableCdfBasis.from_diagrams([a,b],B,23,37,0.35,0.72,weight_scale=0.4)
    assert basis.coordinate_counts()==(2,2)
    for d in (a,b):
        np.testing.assert_allclose(
            basis.transform(d),
            stable_persistence_image(d,B,23,37,0.35,0.72,weight_scale=0.4,engine="dense"),
            rtol=1e-12,atol=1e-12,
        )
    with pytest.raises(ValueError):basis.transform([(0.7,2.0)])

@pytest.mark.parametrize("engine",["auto","shared"])
def test_mixed_spacing_empty(engine):
    a=np.array([(0.2,0.5)]*100)
    b=np.array([(1.1,0.1)]*95)
    diag=np.array([(0.2,0.0)]*50)
    ds=[a,b,diag,np.empty((0,2))]
    spacing=[1.0,2.0,1.0,2.0]
    images=stable_batch_images(ds,B,32,25,0.4,0.3,
        spacings=spacing,weight_scale=1.0,engine=engine)
    for i,d in enumerate(ds):
        np.testing.assert_allclose(
            images[i],
            stable_persistence_image(d,B,32,25,0.4*spacing[i],0.3*spacing[i],
                weight_scale=1.0,engine="dense"),rtol=1e-12,atol=1e-12)

def test_coordinate_reuse_but_no_repeated_pairs():
    ds=[
        np.array([(0.1,0.3),(0.2,0.4),(0.3,0.5)]*20),
        np.array([(0.1,0.5),(0.2,0.3),(0.3,0.4)]*20),
    ]
    basis=StableCdfBasis.from_diagrams(ds,B,13,14,0.6,0.4,weight_scale=.75)
    for d in ds:
        np.testing.assert_allclose(basis.transform(d),
            stable_persistence_image(d,B,13,14,.6,.4,weight_scale=.75),
            rtol=1e-12,atol=1e-12)

def test_memory_guard_and_invalid():
    with pytest.raises(ValueError):
        StableCdfBasis.from_diagrams([[(0.1,-0.1)]],B,10,10,.2,.2)
    with pytest.raises(ValueError):
        StableCdfBasis.from_diagrams([[(0.1,0.4)]],B,10,10,.2,.2,max_cells=0)
    basis=StableCdfBasis.from_diagrams([[(0.1,0.4)]],B,10,10,.2,.2)
    with pytest.raises(ValueError):basis.transform([(0.7,0.4)])
