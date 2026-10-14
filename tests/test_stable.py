from __future__ import annotations
import numpy as np
import pytest
from persistence_images import Bounds, stable_persistence_image, stable_batch_images

@pytest.mark.parametrize("nx,ny,sx,sy", [(8,9,0.15,0.2), (37,21,0.4,0.13), (3,4,1,2)])
def test_reference(nx,ny,sx,sy):
    from scipy.stats import norm
    bounds=Bounds(-2,3,-1,4)
    pts=np.array([(0.2,0.6),(1.1,1.8),(-0.5,0)])
    out=stable_persistence_image(pts,bounds,nx,ny,sx,sy).reshape(nx,ny)
    xe=np.linspace(-2,3,nx+1);ye=np.linspace(-1,4,ny+1)
    ref=np.zeros((nx,ny))
    for b,p in pts:
        ref+=min(p,1)*np.outer(np.diff(norm.cdf((xe-b)/sx)),np.diff(norm.cdf((ye-p)/sy)))
    np.testing.assert_allclose(out,ref,atol=2e-12,rtol=1e-11)

def test_diagonal_mass_empty_and_grouping():
    b=Bounds(-10,10,-10,10)
    assert not np.any(stable_persistence_image([(0,0)],b,40,40,1,1))
    assert not np.any(stable_persistence_image([],b,40,40,1,1))
    a=stable_persistence_image([(0,0.2)],b,40,40,1,1)
    assert abs(a.sum()-0.2)<1e-12
    d=[(0.2,0.6)]*18+[(0.3,1.8)]*4
    dense=stable_persistence_image(d,b,40,40,.5,.3,engine="dense")
    grouped=stable_persistence_image(d,b,40,40,.5,.3,engine="factorized")
    np.testing.assert_allclose(dense,grouped,atol=1e-12,rtol=1e-12)

def test_w1_stability_and_validation():
    from math import pi,sqrt
    b=Bounds(-2,2,0,3)
    a=stable_persistence_image([(0.1,0.5)],b,30,28,.25,.33)
    c=stable_persistence_image([(0.1001,0.5002)],b,30,28,.25,.33)
    C=2+sqrt(2/pi)*(1/.25+2/.33)
    assert np.sum(np.abs(a-c)) < C*.0003
    tiny=stable_persistence_image([(.1,.00001)],b,30,28,.25,.33)
    assert tiny.sum()<=C*.000005
    for bad in [[(0.,-.1)],[(0.,float("inf"))]]:
        with pytest.raises(ValueError):stable_persistence_image(bad,b)
    with pytest.raises(ValueError):stable_persistence_image([],Bounds(0,0,0,1))
    with pytest.raises(ValueError):stable_persistence_image([],b,weight_scale=0)

def test_batch_spacing_and_cli(tmp_path):
    import csv
    from persistence_images.cli import main
    b=Bounds(0,3,0,3)
    points=[[(.1,.7)],[(1.1,.5)]]
    result=stable_batch_images(points,b,20,25,.2,.3,spacings=[1,2],weight_scale=.7)
    for i in range(2):
        np.testing.assert_allclose(result[i],stable_persistence_image(
            points[i],b,20,25,.2*[1,2][i],.3*[1,2][i],weight_scale=.7),atol=1e-13)
    inp=tmp_path/"diagrams"; inp.mkdir()
    (inp/"a.csv").write_text("0.25,0.75\\n0.20,0.20\\n")
    (inp/"b.csv").write_text("0.5,1.0\\n")
    spacing=tmp_path/"spacing.csv"
    spacing.write_text("filename,other,spacing\\na.tif,ok,1.0\\nb.tif,ok,2.0\\n")
    outputs=[]
    for flag in ([],["--stream"]):
        out=tmp_path/("stream.csv" if flag else "batch.csv")
        assert main(["--input",str(inp),"--spacing",str(spacing),"--output",str(out),
                     "--representation","stable","--bounds","0","3","0","3",
                     "--weight-scale","1","--nx","16","--ny","12",*flag])==0
        with out.open() as handle: outputs.append(list(csv.reader(handle)))
    np.testing.assert_allclose(
        [[float(x) for x in row[1:]] for row in outputs[0]],
        [[float(x) for x in row[1:]] for row in outputs[1]],atol=1e-13,rtol=0)
