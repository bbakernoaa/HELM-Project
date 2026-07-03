# Spec: AXIS Rectilinear Normal Lat-Lon Named Grids Support (R-Family)

## 1. Overview & Motivation
This specification defines the extension of the `NamedGridRegistry` in AXIS to support a new **`R` family** representing global Rectilinear Normal Lat-Lon grids (e.g., `R360`).

This provides an easy, standardized, on-the-fly analytical generation mechanism for standard lat-lon regular grids without file-I/O dependencies or complex specification files, mirroring the existing Gaussian named grids (`O<N>`, `F<N>`, `N<N>`).

---

## 2. Named Grid Parsing & Suffix Layout
The `NamedGridRegistry::parse` and validation logic is extended to recognize the `'R'` family character prefix:
*   Names starting with `"R"` or `"r"` are mapped to the family prefix `'R'`.
*   The trailing digits are parsed as a positive integer representing the grid resolution parameter $N$.
*   Example: `"R360"` parses into `ParsedName{family: 'R', number: 360}`.
*   Dimensions:
    *   $ni$ (longitudes) = $4 \times N$
    *   $nj$ (latitudes) = $2 \times N$
    *   For $N = 360$, $ni = 1440$ and $nj = 720$, yielding the standard $1440 \times 720$ grid size.

---

## 3. Grid Generation & Integration
The `NamedGridRegistry::generate` dispatcher will build the R-family grid using the existing `generate_regular_grid` utility function:

*   **Grid Sizing & Step Sizes:**
    *   $ni = 4 \times N$
    *   $nj = 2 \times N$
    *   $dlon = 360.0 / ni$
    *   $dlat = 180.0 / nj$
*   **Grid Boundaries:**
    *   Longitude start: $-180.0^\circ$
    *   Latitude start: $-90.0^\circ$
*   **Dispatcher Code:**
    ```cpp
    case 'R': {
        const int N = parsed.number;
        const std::size_t ni = static_cast<std::size_t>(4 * N);
        const std::size_t nj = static_cast<std::size_t>(2 * N);
        const double dlon = 360.0 / static_cast<double>(ni);
        const double dlat = 180.0 / static_cast<double>(nj);
        return generate_regular_grid<Kokkos::HostSpace>(ni, nj, -180.0, -90.0, dlon, dlat);
    }
    ```

---

## 4. Verification & Testing Strategy
*   **Unit Tests:**
    *   Test validation and parsing of the `R` family grid names (e.g., `"R4"`, `"r360"`).
    *   Verify that `NamedGridRegistry::generate<MemSpace>("R4")` yields a non-empty mesh with $ni = 16$ and $nj = 8$ cells (total $128$ cells).
    *   Verify that coordinates span exactly $[-180.0, 180.0]$ in longitude and $[-90.0, 90.0]$ in latitude.
*   **Property & Determinism Tests:**
    *   Ensure that multiple calls to `"R360"` or other R-family grids produce bitwise-identical meshes, adhering to standard determinism requirements.
