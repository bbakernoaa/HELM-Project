# Spec: AXIS Spherical Unit-Vector Centroid Calculations (Gap D)

## 1. Overview & Motivation
Currently, unstructured cell centroids in AXIS (`compute_cell_centroids_xy` on Host and `compute_cell_centroids_device` on Device) are calculated by taking the simple average of the physical node coordinates $(lon, lat)$ of each cell.

However, near the geographical dateline crossing ($180^\circ$ or $360^\circ$ longitude), coordinate wrap-around causes severe centroid calculations errors. For example, a cell crossing $360^\circ$ longitude with nodes at $359^\circ$ and $1^\circ$ yields an incorrect average centroid longitude of $180^\circ$ (on the exact opposite side of the globe), breaking nearest-neighbor matching and bilinear stencils near the boundaries.

This specification defines the implementation of a robust **3D Unit-Vector Spherical Centroid Calculator** in C++ for both Host and Device coordinate computations.

---

## 2. Mathematical Formulation

To calculate the centroid of a cell with $V$ vertices under a spherical coordinate system (degrees or radians):

1.  **Coordinate Conversion to 3D Cartesian Space:**
    For each vertex $i$ of the cell, convert geographical coordinates $(\lambda_i, \theta_i)$ (in radians) to its unit-vector representation $(X_i, Y_i, Z_i)$ on the sphere:
    $$X_i = \cos\theta_i \cos\lambda_i$$
    $$Y_i = \cos\theta_i \sin\lambda_i$$
    $$Z_i = \sin\theta_i$$

2.  **3D Vector Averaging:**
    Compute the average 3D unit-vector $(\bar{X}, \bar{Y}, \bar{Z})$ across all vertices:
    $$\bar{X} = \frac{1}{V}\sum_{i=1}^{V} X_i, \quad \bar{Y} = \frac{1}{V}\sum_{i=1}^{V} Y_i, \quad \bar{Z} = \frac{1}{V}\sum_{i=1}^{V} Z_i$$

3.  **Inverse Spherical Conversion:**
    Convert the averaged unit-vector back to geographical centroid $(\bar{\lambda}, \bar{\theta})$:
    $$\bar{\theta} = \arcsin\bar{Z}$$
    $$\bar{\lambda} = \text{atan2}(\bar{Y}, \bar{X})$$
    If $\bar{\lambda} < 0$, wrap to $[0, 2\pi)$ by adding $2\pi$.

4.  **Scaling:**
    Convert $(\bar{\lambda}, \bar{\theta})$ back to degrees if the grid coordinate system is `SphericalDeg`.

---

## 3. Verification & Testing Strategy

*   **Unit Tests:**
    *   Verify that calculating the centroid of a cell spanning across the $360^\circ$ dateline discontinuity yields the correct physical centroid close to $0^\circ / 360^\circ$ rather than $180^\circ$.
*   **Property-Based & Integration Tests:**
    *   Ensure all existing test suites compile and pass clean, demonstrating full compatibility and bitwise-identical centroid results for non-crossing cells.
