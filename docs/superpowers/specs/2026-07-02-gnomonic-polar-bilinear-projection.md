# Spec: AXIS Gnomonic Tangent-Plane Polar Projection for Bilinear Interpolation (Gap C)

## 1. Overview & Motivation
In unstructured bilinear interpolation (`WeightGenerator::generate_bilinear`), physical coordinates $(lon, lat)$ are mapped to element reference coordinates $(\xi, \eta)$ using Newton iteration (`map_to_reference_quad`) or barycentric coordinates (`barycentric_triangle`).

However, near the geographic poles ($\pm 90^\circ$ latitude), the extreme meridian convergence causes severe coordinate compression and singularities in longitude. This leads to Newton iteration divergence, dividing by zero, or failure to locate points inside cells.

This specification defines the implementation of a **Local Gnomonic Tangent-Plane Projection** for spherical unstructured bilinear weight generation on the host. This projects the localized cell vertices and destination points onto a conformal, flat, distortion-free tangent plane centered at the destination point before calculating interpolation weights.

---

## 2. Mathematical Formulation

For any destination point $P = (\lambda_0, \theta_0)$ (longitude, latitude in radians) and any source point $S = (\lambda, \theta)$:

1.  **Tangent Plane Center:**
    We set the projection center at the destination point $P$. This means $P$ projects exactly to $(0, 0)$ in the tangent plane.

2.  **Gnomonic Projection Equations:**
    For any source vertex $S = (\lambda, \theta)$, its projected coordinates $(u, v)$ on the tangent plane are:
    $$\cos c = \sin\theta_0 \sin\theta + \cos\theta_0 \cos\theta \cos(\lambda - \lambda_0)$$
    $$u = \frac{\cos\theta \sin(\lambda - \lambda_0)}{\cos c}$$
    $$v = \frac{\sin\theta \cos\theta_0 - \cos\theta \sin\theta_0 \cos(\lambda - \lambda_0)}{\cos c}$$

3.  **Local Cartesian Interpolation:**
    Once the destination point $(0, 0)$ and the cell vertices $(u_i, v_i)$ are projected, we run the reference mapping:
    *   For quads: Solve `map_to_reference_quad(0.0, 0.0, {u0, v0}, {u1, v1}, {u2, v2}, {u3, v3}, xi, eta)`.
    *   For triangles: Solve `barycentric_triangle(0.0, 0.0, {u0, v0}, {u1, v1}, {u2, v2}, l0, l1, l2)`.

This completely eliminates polar coordinate singularities and ensures perfect numerical convergence near the poles.

---

## 3. Verification & Testing Strategy

*   **Unit Tests:**
    *   Add a test case in `test_axis_regridder` (or C++ unit tests) evaluating bilinear interpolation of a quad cell located at $89.9^\circ$ latitude.
    *   Verify that the Newton iteration converges perfectly to a relative tolerance of $1.0 \times 10^{-12}$.
