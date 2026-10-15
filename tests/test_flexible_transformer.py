"""Flexible kernels, scikit-learn integration, and visualization regressions."""
import numpy as np
import pytest

from persistence_images import (
    Bounds, PersistenceImageTransformer, flexible_persistence_image,
    stable_persistence_image, persistence_weights, plot_diagram, plot_image
)

def test_diagonal_reference_and_custom_ramp():
    b=Bounds(-1,2,-1,3)
    points=np.array([[0.1,0.3],[0.1,0.0],[0.7,1.1]])
    ref=stable_persistence_image(points,b,17,13,0.2,0.38,
                                 weight_scale=0.8,engine="dense")
    candidate=flexible_persistence_image(
        points,b,17,13,0.2,0.38,weight="ramp",weight_scale=0.8)
    np.testing.assert_allclose(candidate,ref,rtol=1e-12,atol=1e-12)

def test_custom_weight_and_covariance_diagonal():
    b=Bounds(0,2,0,2)
    points=np.array([[0.2,0.3],[1.1,0.4]])
    got=flexible_persistence_image(
        points,b,13,12,weight=lambda p: p*p,
        covariance=[[0.04,0.0],[0.0,0.09]])
    a=flexible_persistence_image(
        points,b,13,12,weight="power",weight_params={"exponent":2},
        covariance=[[0.04,0.0],[0.0,0.09]])
    np.testing.assert_allclose(got,a,atol=1e-13,rtol=1e-13)
    assert got.shape==(13*12,)
    assert np.all(got>=0.0)

def test_correlated_covariance_against_reference():
    scipy=pytest.importorskip("scipy")
    from scipy.stats import multivariate_normal
    b=Bounds(-.4,.6,-.2,.7)
    mean=np.array([0.13,0.21])
    covariance=np.array([[.08,.025],[.025,.06]])
    image=flexible_persistence_image(
        [mean],b,3,3,weight="ramp",weight_scale=1.0,
        covariance=covariance,quadrature_nodes=96).reshape(3,3)
    xe=np.linspace(b.x_min,b.x_max,4)
    ye=np.linspace(b.y_min,b.y_max,4)
    rv=multivariate_normal(mean=mean,cov=covariance)
    for i in range(3):
        for j in range(3):
            expected=(rv.cdf([xe[i+1],ye[j+1]])
                      -rv.cdf([xe[i],ye[j+1]])
                      -rv.cdf([xe[i+1],ye[j]])
                      +rv.cdf([xe[i],ye[j]]))*mean[1]
            assert abs(image[i,j]-expected)<2e-8

def test_rejected_weights_and_covariance():
    p=np.array([.1,.4])
    assert np.allclose(persistence_weights(p,"linear_ramp",
            weight_params={"start":.2,"end":.4}),[0,1])
    for bad in [lambda x:-np.ones(len(x)),lambda x: np.array([1,2,3])]:
        with pytest.raises(ValueError):
            persistence_weights(p,bad)
    with pytest.raises(ValueError):
        flexible_persistence_image([[.1,.3]],Bounds(0,1,0,1),
            covariance=[[1,2],[2,1]])
    with pytest.raises(ValueError):
        persistence_weights(p,"power",weight_params={"exponent":.4})

def test_estimator_fitted_bounds_and_diagrams():
    data=[np.array([[.1,.4],[.3,.7]]),
          np.array([[.2,.2]]),
          np.empty((0,2))]
    t=PersistenceImageTransformer(nx=11,ny=9,padding=.12)
    X=t.fit_transform(data)
    assert X.shape==(3,99)
    assert np.all(X[2]==0)
    assert len(t.get_feature_names_out())==99
    bounds=t.bounds_
    heldout=[np.array([[8.,8.]])]
    assert t.transform(heldout).shape==(1,99)
    assert t.bounds_ == bounds
    assert t.transform_images(data).shape==(3,11,9)
    with pytest.raises(RuntimeError):
        PersistenceImageTransformer().transform(data)

def test_estimator_coordinates_and_custom_reference():
    bp=np.array([[.2,.4],[.1,.5]])
    bd=np.column_stack((bp[:,0],bp.sum(axis=1)))
    b=Bounds(0,2,0,2)
    a=PersistenceImageTransformer(bounds=b,nx=8,ny=7,
        coordinates="birth_death").fit_transform([bd])
    r=stable_persistence_image(bp,b,8,7,.2,.2)
    np.testing.assert_allclose(a[0],r,atol=1e-13,rtol=1e-12)
    custom=PersistenceImageTransformer(bounds=b,nx=8,ny=7,
        weight="power",weight_params={"exponent":2}).fit_transform([bp])
    assert custom.shape==(1,56)
    with pytest.raises(ValueError):
        PersistenceImageTransformer(bounds=b,engine="interp_fft",
            weight="power").fit_transform([bp])

def test_sklearn_clone_pipeline():
    sklearn=pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    diagrams=[
        np.array([[.1,.15],[.2,.21]]) if k%2==0 else
        np.array([[.1,.71],[.3,.81]]) for k in range(8)]
    labels=np.array([k%2 for k in range(8)])
    estimator=PersistenceImageTransformer(nx=8,ny=8,weight_scale=.5)
    cloned=clone(estimator)
    assert cloned.get_params()["nx"]==8
    pipe=make_pipeline(estimator,LogisticRegression(max_iter=400))
    pipe.fit(diagrams,labels)
    assert len(pipe.predict(diagrams))==8

def test_plotting_optional():
    mpl=pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    b=Bounds(0,1,0,1)
    ax=plot_diagram([[.2,.4]],coordinates="birth_persistence")
    assert ax.get_ylabel()=="Persistence"
    plt.close(ax.figure)
    image=stable_persistence_image([[.2,.4]],b,7,5,.2,.3)
    ax=plot_image(image,b,7,5,colorbar=False)
    assert ax.get_xlabel()=="Birth"
    plt.close(ax.figure)
