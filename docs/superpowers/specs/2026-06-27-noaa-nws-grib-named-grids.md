# Spec: AXIS NOAA NWS Named GRIB Grids Support (G-Family)

## 1. Overview & Motivation
Meteorological models operated by NOAA NWS (National Weather Service) and NCEP (National Centers for Environmental Prediction) heavily utilize standardized grids identified by official GRIB grid numbers (such as `grid3`, `grid4`, or `grid218`).

This specification defines the architectural and interface designs to add a new **`G` family** of named grids to the `NamedGridRegistry`, allowing these standard grids to be generated dynamically and analytically on-the-fly inside AXIS without any heavy file-I/O dependencies.

---

## 2. Named Grid Parsing & Suffix Layout
The `NamedGridRegistry::parse` function is extended to recognize the `"grid<number>"` pattern:
*   Names starting with `"grid"` (case-insensitive) are mapped to family prefix `'G'`.
*   The trailing digits are parsed as a positive integer grid number.
*   Example: `"grid218"` parses into `ParsedName{family: 'G', number: 218}`.

---

## 3. Registered NOAA GRIB Grid Generators

The `NamedGridRegistry::generate` dispatcher is extended to build G-family grids based on NCEP/NOAA grid definitions:

### A. `"grid3"` (GFS 1.0° Global Grid)
*   **Dimensions:** $ni = 360$, $nj = 181$.
*   **Layout:** Regular Gaussian/regular lat-lon.
*   **Grid bounds:** Longitudes spanning $0.0^\circ$ to $360.0^\circ$ ($dlon = 1.0$), latitudes spanning $-90.0^\circ$ to $90.0^\circ$ ($dlat = 1.0$).

### B. `"grid4"` (GFS 0.5° Global Grid)
*   **Dimensions:** $ni = 720$, $nj = 361$.
*   **Layout:** Regular Gaussian/regular lat-lon.
*   **Grid bounds:** Longitudes spanning $0.0^\circ$ to $360.0^\circ$ ($dlon = 0.5$), latitudes spanning $-90.0^\circ$ to $90.0^\circ$ ($dlat = 0.5$).

### C. `"grid218"` (NAM / RAP 12km ConUS Regional Grid)
*   **Dimensions:** $ni = 614$, $nj = 428$.
*   **Layout:** Curvilinear structured grid.
*   **LCC Projection Parameters:**
    *   `+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=-95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs`
    *   Centers are generated in projection space: $x$ from $-3,733,392$m to $3,733,392$m ($dx = 12,191$m), and $y$ from $-2,602,779$m to $2,602,779$m ($dy = 12,191$m).
    *   **PROJ Dependency:** Requires compiling AXIS with PROJ enabled (`AXIS_ENABLE_PROJ=ON`). If PROJ support is disabled, requesting `"grid218"` throws a descriptive `std::runtime_error` explaining that PROJ is required to generate projected regional grids.

---

## 4. Verification & Testing Strategy
*   **Grid3/4 Dimension and Bounds Tests:** Asserts that `"grid3"` and `"grid4"` generate correct dimensions ($360 \times 181$ and $720 \times 361$, respectively) and span the exact geographic global boundary ranges.
*   **Grid218 LCC Projection Tests:** When PROJ is enabled, verifies that `"grid218"` generates exactly $614 \times 428 = 262,792$ cells, and that center coordinates are converted to valid geographic lon/lat.
*   **G-Family Parsing Tests:** Verifies that `"grid"` names with invalid suffixes (e.g. `"gridABC"` or `"grid0"`) are caught and rejected with descriptive parsing errors.
