# Certified integrated one-FFT persistence images

## Definition and discretization

For a diagram D of birth–persistence points (b,p), p >= 0, use the bounded
Lipschitz ramp w(p) = min(1,p/s), s > 0, and Gaussian *probability* density
g_sigma(t) = exp(-t²/(2 sigma²))/(sqrt(2 pi) sigma). For a common fixed grid
of rectangles R_ij, fixed positive bandwidths, and a fixed ramp scale, the
stable feature is

    I_ij(D) = sum_{(b,p) in D} w(p) integral_Rij
               g_sigma_x(x-b) g_sigma_y(y-p) dx dy.

It is the integral, not the value at a pixel center. The image is not
normalized by its own mass.

For pixel-center lattice spacing (h_x,h_y), tensor-product Lagrange
interpolation replaces the source at (b,p) by signed neighboring lattice
weights. Deposit w(p) times these interpolation coefficients, then convolve
once with the integrated Gaussian lattice kernel:

    K_ab = [Phi((a+1/2)*h_x/sigma_x)-Phi((a-1/2)*h_x/sigma_x)]
           [Phi((b+1/2)*h_y/sigma_y)-Phi((b-1/2)*h_y/sigma_y)].

The implementation uses complementary CDF differences in the tails and
pads the real FFT enough to avoid periodic wraparound at output pixels.

## Per-pixel error theorem

Let W(D) = sum w(p), and let 2r be the even stencil width. The maximum
absolute product of distances to the 2r equally spaced central nodes is

    ((2r)!)² * h^(2r) / (2^(4r) * (r!)²).

For one Gaussian pixel mass regarded as a function of its source position,
the 2r-th derivative has the uniform bound

    h * E|Z|^(2r) / (sqrt(2*pi) * sigma^(2r+1)),

with Z a standard normal variable. Applying Lagrange interpolation with
the derivative divided by (2r)! gives

    e_r(h,sigma) =
      (h/(sqrt(2*pi)*sigma))
      * ((2r)!)²/(2^(5r)*(r!)³) * (h/sigma)^(2r).

Separability, Gaussian pixel masses between zero and one, and the tensor
product error decomposition imply the uniform bound

    ||I(D)-I_approx(D)||_infinity
       <= W(D) * (e_x + e_y + e_x*e_y).

This bounds interpolation in *exact arithmetic*. It does not guarantee
FFT/deposition rounding. Widths are selected to satisfy this bound; when
none within 18 nodes per axis works, the explicit FFT engine rejects the
requested tolerance.

## Entire-feature L1 error theorem

In the Banach space L1(R), the 2r-th derivative in source position of a
normalized Gaussian has norm at most sqrt((2r)!)/sigma^(2r), by the Hermite
polynomial identity and Cauchy–Schwarz. The Hermite–Genocchi divided
difference remainder and the same node product yield the one-axis bound

    B_r(h,sigma) =
       ((2r)!)^(3/2)/(2^(4r)*(r!)²) * (h/sigma)^(2r).

Gaussian densities each have total L1 mass one. Pixel integration over
disjoint rectangles is L1-contractive. Thus the vector error obeys

    ||I(D)-I_approx(D)||_1
       <= W(D) * (B_x + B_y + B_x*B_y).

Post-FFT clipping to nonnegative values cannot increase either bound
relative to exact nonnegative pixel integrals.

## Relationship to Wasserstein stability

For an exact stable image map with a fixed common grid, bandwidths and
ramp scale, suppose ||I(D)-I(E)||_1 <= C*W1(D,E). If approximate images have
vector errors eta_D and eta_E as above, the triangle inequality gives

    ||I_approx(D)-I_approx(E)||_1
       <= C*W1(D,E) + eta_D + eta_E.

This is additive-error approximate stability, NOT a proof that the
approximating algorithm is exactly Lipschitz under W1.

## Use and restrictions

Pass representation=stable, engine=interp_fft, absolute_error=1e-8 on the
CLI, or call stable_interp_fft_image(...,return_certificate=True). A fixed
grid is crucial for comparison. Explicit FFT requires all source coordinates
within the fixed physical grid. Stable auto can fall back to exact CDF
integration for out-of-grid sources or under-resolved/narrow Gaussians.
Default stable computation still performs no intentional approximation.

Runtime approximately O(N*w_x*w_y + P log P) with P the padded FFT size
and a finite memory cap. One FFT kernel can be reused when pixel widths,
resolution, bandwidths, and stencil widths are identical.

The Gaussian FFT gridding strategy is not claimed to be novel relative to
the numerical-analysis literature. The mathematical target here is the
classical normalized, weighted, pixel-integrated persistence image.
