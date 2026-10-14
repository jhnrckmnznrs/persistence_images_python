# Error-certified one-FFT persistence images

## Setting

Given points `z_k=(b_k,p_k)` in a common rectangular domain and target pixel centers
`t_ij=(x_i,y_j)`, compute the *unnormalized point-sampled* image

\[I_{ij}=\sum_{k=1}^N G_x(x_i-b_k)G_y(y_j-p_k), \qquad G_x(u)=\exp(-u^2/(2\sigma_x^2)),\quad G_y(v)=\exp(-v^2/(2\sigma_y^2)).\]

No persistence weighting, pixel integration, or Gaussian normalizing factors are used.
Let the target-lattice spacings be `h_x`, `h_y`. Extend the lattice with enough
**ghost** nodes so that every source point has a central interpolation stencil.

## Theorem: one-convolution approximation and explicit certificate

For each source `k`, select `w_x` consecutive lattice centers nearest its birth
coordinate and `w_y` nearest its persistence coordinate. Let `L_{ka}(b)` and
`R_{kb}(p)` be the corresponding univariate cardinal Lagrange interpolation
weights, evaluated at `(b_k,p_k)`. Construct a *signed* gridded mass distribution

\[\mu_{ab}=\sum_{k=1}^N L_{ka}(b_k)R_{kb}(p_k),\]

with the understanding that `L,R` vanish outside the point's local stencil.
Form one Gaussian lattice convolution:

\[\widetilde I_{ij}=\sum_{a,b}\mu_{ab}G_x(x_i-x_a)G_y(y_j-y_b).\]

Let

\[c_w=\frac{\mathbb E|Z|^w}{w!}=\frac{2^{w/2}\Gamma((w+1)/2)}{\sqrt\pi\,w!},\quad Z\sim\mathcal N(0,1),\]

and define per-source quantities

\[e_{xk}=c_{w_x}\prod_{a\in S_{xk}}\frac{|b_k-x_a|}{\sigma_x},\qquad
 e_{yk}=c_{w_y}\prod_{b\in S_{yk}}\frac{|p_k-y_b|}{\sigma_y}.\]

Then, *in exact arithmetic*,

\[\boxed{\|I-\widetilde I\|_\infty\ \leq\ \sum_{k=1}^N\left(e_{xk}+e_{yk}+e_{xk}e_{yk}\right).}\]

**Proof.** Gaussian characteristic-function representation yields
`G_x(t-b) = E[exp(i Z(t-b)/sigma_x)]` and hence
`sup_b |d^w G_x(t-b)/db^w| <= E|Z|^w / sigma_x^w`, uniformly in `t`.
Lagrange's remainder gives `|G_x(t-b_k)-I_xG_x(t-b_k)| <= e_xk`, and similarly in `y`.
Writing `f_x f_y-\tilde f_x\tilde f_y=(f_x-\tilde f_x)f_y+\tilde f_x(f_y-\tilde f_y)`
and using `|f_x|,|f_y|<=1` and `|\tilde f_x|<=1+e_x` gives
`<= e_x+e_y+e_xe_y`. Sum over sources. QED.

## Corollary: sharp uniform certificate for *even* central stencils

Take `w=2r` neighboring nodes in one dimension, with the source located between
the two central nodes. For any such point, its dimensionless product obeys

\[\prod_{a=1}^{2r}\frac{|b-x_a|}{\sigma} \leq
 \left(\frac{h}{\sigma}\right)^{2r}
 \left[\frac{(2r)!}{4^r r!}\right]^2.\]

**Proof.** Put the coordinate of the source within the central lattice cell
at `u` with `0<=u<=1`. The product equals
`u(1-u) ∏_{j=1}^{r-1} [j(j+1)+u(1-u)] * h^(2r)`.
Every factor is increasing with `u(1-u)`, so the maximum is attained at
`u=1/2`. The resulting product is
`[∏_{j=0}^{r-1}(j+1/2)]^2 h^(2r)`.

For even `w=2r`, `c_w=1/(2^r r!)`. Consequently the one-dimensional bound is

\[\boxed{E_r(h,\sigma)=
\frac{[(2r)!]^2}{2^{5r}(r!)^3}\left(\frac{h}{\sigma}\right)^{2r}.}\]

A globally uniform and data-independent (apart from `N`) image certificate is

\[\boxed{\|I-\widetilde I\|_\infty\ \leq\ N\left(E_{r_x}(h_x,\sigma_x)+E_{r_y}(h_y,\sigma_y)+E_{r_x}(h_x,\sigma_x)E_{r_y}(h_y,\sigma_y)\right).}\]

In particular, stencil widths can be chosen in **constant time** by minimizing
`w_x*w_y` over a small menu of even widths meeting the chosen tolerance.
This avoids examining all the source displacements prior to gridding.

The ratio of consecutive one-dimensional *uniform bounds* is exactly

\[\frac{E_{r+1}}{E_r}=\frac{(2r+1)^2}{8(r+1)}\left(\frac{h}{\sigma}\right)^2.\]

This makes explicit why increasing the stencil width eventually becomes
counterproductive at fixed grid resolution: the stencil grows wider while
its interpolation factors grow factorially. Narrow kernels on coarse grids
may therefore have **no certified width** in the supported range, and should
fall back to localized evaluation or dense summation.

## Complexity and memory

Depositing each of `N` sources onto a `w_x * w_y` stencil costs
`O(N w_x w_y)`. The image is obtained by one zero-padded FFT convolution
costing `O(P log P)` for `P` FFT cells. A cached Gaussian kernel spectrum
is reusable across diagrams of identical grid/bandwidth. `O(P)` additional
workspace is needed apart from input/output and stencil scratch buffers.

The previous degree-`q` moment method has
`(q+1)(q+2)/2` moment arrays and as many FFT convolutions, with cost
`O(N q^2 + q^2 P log P)` and `O(q^2 P)` cached spectra. Local interpolation
reduces the FFT count to 1, at the expense of `w_x*w_y` deposition work.

## Floating-point caution

The theorem bounds **interpolation error in exact arithmetic**. It does *not*
bound rounding errors in the barycentric stencil, the signed-mass deposition,
or the FFT convolution. Floating-point summation order differs from Rust's
nested loops. Extremely strict tolerances should be checked against a direct
or high-precision reference. Widths larger than approximately 18 may need
special care with cancellation and weight stability. Theoretical inequalities
are not a proof of implementation-level bitwise equality.

## Scientific status

The source-gridding idea is related to particle-mesh interpolation, smooth-kernel
summation, and NUFFT spreading; the claim is a **specialized, rigorous uniform
certificate** for the particular unnormalized anisotropic Gaussian feature map
and a measured engineering improvement. It has not been established that the
result is novel relative to all existing fast Gaussian and NUFFT literature.

Relevant literature includes: Greengard and Strain (1991), Greengard et al.
(SIAM Review 2024, DOI 10.1137/23M1572453), Barnett (Applied and Computational
Harmonic Analysis 2021, DOI 10.1016/j.acha.2020.10.002), and Barnett,
Greengard, Rachh (2023, arXiv 2305.11065) on uniform Fourier-grid
approximation of squared-exponential kernels.
