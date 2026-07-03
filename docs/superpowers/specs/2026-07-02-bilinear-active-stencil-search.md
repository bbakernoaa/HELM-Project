# Spec: AXIS Bilinear Active-Stencil Search Expansion (Gap H)

## 1. Overview & Motivation
Currently, unstructured bilinear interpolation (`WeightGenerator::generate_bilinear`) queries exactly the $k=4$ nearest source centroids. It tests if the destination cell centroid is inside the quad or triangle formed by these 4 closest centroids. If the point is outside (which is common near grid boundaries, complex coastlines, or irregular grids), the library falls back directly to Inverse-Distance Weighting (IDW).

This specification defines the implementation of an **Active-Stencil Search Expansion (Gap H)**. By expanding the neighborhood search radius to $k=8$ nearest centroids and dynamically evaluating combinations of 4-point (quad) or 3-point (triangle) stencils, we dramatically increase the probability of locating an exact bounding stencil, reducing boundary blending artifacts and optimizing interpolation accuracy.

---

## 2. Design & Stencil Search Algorithm

1.  **Expanded Neighborhood Query:**
    *   Increase the ArborX nearest neighbor search count from $k=4$ to $k=8$:
        ```cpp
        const int k_query = static_cast<int>(std::min(static_cast<std::size_t>(8), n_src));
        ```
    *   ArborX will return up to 8 nearest centroids ordered by distance for each destination cell.

2.  **Dynamic Stencil Evaluation Loop:**
    For each destination cell $j$ with $A$ available neighbor centroids ($A \le 8$):
    *   **Phase A: Convex Quad Search:**
        We evaluate combinations of 4 centroids chosen from the $A$ neighbors, prioritized by closeness (e.g. testing combinations involving the closest neighbors first). For each candidate quad combination $\{c_a, c_b, c_c, c_d\}$:
        *   Project coordinates to local tangent plane centered at the destination point.
        *   Check if $(0.0, 0.0)$ is inside the quad and the quad is convex.
        *   If valid, solve `map_to_reference_quad`, assemble the 4 bilinear weights, and exit search successfully.
    *   **Phase B: Triangle Search:**
        If no enclosing quad is found, we evaluate combinations of 3 centroids chosen from the $A$ neighbors. For each candidate triangle combination $\{c_a, c_b, c_c\}$:
        *   Project coordinates to local tangent plane.
        *   Check if $(0.0, 0.0)$ is inside the triangle.
        *   If valid, solve `barycentric_triangle`, assemble the 3 barycentric weights, and exit search successfully.
    *   **Phase C: IDW Fallback:**
        If no enclosing quad or triangle is found among any combinations of neighbors, fall back to Inverse-Distance Weighting (IDW) using the 4 closest centroids.

---

## 3. Verification & Testing Strategy

*   **Unit Tests:**
    *   Verify that unstructured bilinear interpolation near boundaries produces exact shape-function weights instead of falling back to IDW.
*   **Property-Based & Performance Tests:**
    *   Verify that the full C++ test suite and Python test suite compile clean, run flawlessly, and all 396 tests are 100% green.
