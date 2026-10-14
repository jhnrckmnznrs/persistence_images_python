# Stable weighted, integrated persistence images

## Definition

For a finite birth–persistence diagram `D` with points `(b,p)`, `p>=0`,
fix a nondegenerate axis-aligned pixel grid with cells `R_ij`, strictly
positive standard deviations `sigma_x,sigma_y`, and a persistence
saturation scale `s>0`. Let `w(p)=min(1,p/s)`.

Define the normalized anisotropic Gaussian kernel

```text
g_(b,p)(x,y) = exp(-[(x-b)^2/sigma_x^2 + (y-p)^2/sigma_y^2]/2)
                / (2*pi*sigma_x*sigma_y).
```

The stable persistence image is

```text
PI(D)[i,j] = sum_(b,p) in D w(p) * integral_(R_ij) g_(b,p)(x,y) dx dy.
```

For pixel boundaries `[x_i,x_(i+1)]` and `[y_j,y_(j+1)]`, the integral
factorizes into products of CDF differences:

```text
[Phi((x_(i+1)-b)/sigma_x) - Phi((x_i-b)/sigma_x)]
*
[Phi((y_(j+1)-p)/sigma_y) - Phi((y_j-p)/sigma_y)].
```

No quadrature is required for axis-aligned Gaussian covariance. Gaussian
CDF differences in the positive tail are evaluated by survival-function
symmetry to avoid subtracting numbers rounded to 1.

The Gaussian density has integral one over the plane. The **output image
must not be divided by its own sum**: near the empty diagram this would
destroy continuity. The finite window truncates Gaussian mass.

## Theorem: global Wasserstein-1 stability

Suppose the grid bounds, Gaussian bandwidths and weight scale are **fixed
identically** for the diagrams being compared. The diagram matching
distance `W1_inf` uses the standard birth-death coordinates, the
L-infinity cost and diagonal matches. Then

```text
||PI(D)-PI(E)||_(l1) <= C * W1_inf(D,E)
C = 2/s + sqrt(2/pi) * (1/sigma_x + 2/sigma_y).
```

### Proof

Let `g_z` be the normalized Gaussian centered at `z=(b,p)`. The
derivatives with respect to its center satisfy

```text
||partial_b g_z||_(L1) = sqrt(2/pi)/sigma_x,
||partial_p g_z||_(L1) = sqrt(2/pi)/sigma_y.
```

Integrating along axis-parallel translations gives

```text
||g_z-g_z'||_(L1) <= sqrt(2/pi)
                 * (|b-b'|/sigma_x + |p-p'|/sigma_y).
```

The ramp obeys `0<=w<=1`, `w(0)=0`, and
`|w(p)-w(p')|<=|p-p'|/s`. Therefore

```text
||w(p)g_z - w(p')g_z'||_(L1)
 <= |p-p'|/s
    + sqrt(2/pi)*(|b-b'|/sigma_x+|p-p'|/sigma_y).
```

For a matched pair in birth-death coordinates, writing
`delta=max(|b-b'|,|d-d'|)`, we have `|b-b'|<=delta` and
`|p-p'|<=2delta`, yielding `C*delta`.

A point of persistence `p` matched to the diagonal has cost `p/2`.
Its unmatched weighted density has L1 norm `w(p)<=p/s`, which is at
most `(2/s)*(p/2)`. Thus all matching contributions have the stated
bound. Summing and taking an infimum proves the bound for the weighted
Gaussian **surfaces**.

Finally, the pixels are disjoint, and integration is L1-contractive:

```text
sum_ij |integral_(R_ij) (rho_D-rho_E)|
 <= integral_(union R_ij) |rho_D-rho_E|
 <= ||rho_D-rho_E||_(L1).
```

This proves the image result.

## Preconditions and limitations

- Fixed grid, sigmas, and weight scale are essential. Fitting new bounds for
  every diagram or rescaling bandwidths per sample changes the map.
- The theorem concerns exact integrals, not machine floating-point errors.
- Persistence must be finite and nonnegative. Infinite/essential classes
  need to be removed or represented separately before transformation.
- The ramp and normalized Gaussian are classical choices consistent with
  the framework of Adams et al. (JMLR 2017), including its published erratum.
  This derivation is **not claimed as a new theorem**.
- The historical sampled mode is unweighted and unnormalized and evaluates
  Gaussian values at pixel **centers**. It remains available but is not the
  weighted pixel-integrated representation analyzed here.
