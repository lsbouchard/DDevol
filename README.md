# DDevol

Matrix-free Pauli propagation for dipolar spin dynamics, with independent
density-matrix and Lanczos references, exact moment calculations, and
finite-precision FID bounds.

The numerical engines and analysis programs are C17/C++17. Python is used only
to render precomputed CSV tables. This source distribution contains no binary
executables, numerical output, checkpoints, manuscript files, or development
history.

## Build

Requirements: CMake 3.20+, a C17/C++17 compiler, POSIX threads, GMP with its C++
bindings, MPFR, and BLAS/LAPACK. GCC or Clang must support `__uint128_t`.
Do not enable fast-math or floating-point contraction: rounding assumptions
are part of the bounded calculation.

On Apple Silicon, install the command-line developer tools and the dependencies:

```sh
brew install cmake gmp mpfr
make build
```

The build uses Accelerate for BLAS/LAPACK and builds the Metal backend on macOS.
The Metal kernel performs integer key/sign transformations; coefficient
arithmetic remains on the CPU. POSIX-thread CPU execution is also available.

On Debian/Ubuntu Linux:

```sh
sudo apt-get install build-essential cmake libgmp-dev libmpfr-dev libopenblas-dev liblapack-dev
make build
```

The Linux build enables OpenMP when available, in addition to POSIX threads.
To build the CUDA branch backend with an installed NVIDIA toolkit:

```sh
make build BUILD="$PWD/build-cuda" CMAKE_OPTIONS='-DTPGP_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native'
```

Architecture auto-detection requires a sufficiently recent CMake; alternatively
pass the explicit architecture supported by the installed toolkit and device.
The CUDA backend computes binary64 local branches. The translation-orbit engine
used for the bounded large-ring FID is a CPU implementation, not a CUDA solver.

`make build` also runs the native test suite. GPU tests report a skip if the
corresponding device/backend is unavailable. A skipped test is not a GPU pass.

## Reproduce the calculations

Build first, then select a campaign. Output is created under `results/`.
Each measured process writes a CSV, a diagnostic log, and a resource table
containing elapsed seconds, peak resident bytes, and exit status. Existing
outputs are not overwritten; use a new `RESULTS` directory for a fresh run.

```sh
make core       # Dense comparisons, analytic cases, XY, moments, Lanczos
make long       # N=1,000 uniform-ring refinement and reversal calculations
make finite     # Circumference comparisons
make cubic      # Three-dimensional dipolar examples and lattice moments
make short-bounds # Component/residual bounds and independent small-system checks
make bounded    # Bounded J1-J2 FIDs at N=16 and N=1,000; N=16 full trace
make cost       # Single-thread matched-target comparisons at N=8,16,17
make analyze    # Native numerical comparisons and feature extraction
```

The complete sequence, including figures, is `make reproduce`. Large campaigns
can require many hours and substantial RAM; run them on a persistent compute
host. No time limit is imposed. Resource or error-target stops return nonzero
status and preserve completed outputs. They are not treated as complete runs.

For separate output/build locations:

```sh
make core BUILD="$PWD/build" RESULTS="$PWD/results-new"
```

The finite-precision component/residual examples and mathematical checks can
also be run individually:

```sh
build/test_reference_rev
build/check_cluster_rev
build/check_moments_rev
build/tpgp_rev --help
```

The full test commands and their input parameters are registered in
`engine/CMakeLists.txt` and the native test sources.

## Figures and analyses

Install the plotting dependencies in a Python environment, then render:

```sh
python3 -m pip install -r plots/requirements.txt
make figures
```

The script writes PDF and PNG versions. Separately included panels use equal
canvas dimensions. Symbols are staggered along coincident curves without
moving their numerical coordinates. All differences, intervals, extrapolated
values, and moment envelopes are computed by native programs before plotting.

| Manuscript figure | Data-generating campaigns | Output basename |
|---|---|---|
| 1: removal allowance | `core` | `fig_truncation_bound` |
| 2: bounded FID and full trace | `bounded` | `fig_fid_t2_rev` |
| 3: isolated pair and complete graph | `core` | `fig_pair_analytic`, `fig_uniform_complete_analytic` |
| 4: Ising and XY | `core` | `fig_ising_analytic`, `fig_xy_chain` |
| 5: moment enclosures | `core`, `long`, `analyze` | `fig_moment_bounds` |
| 6: independent Lanczos | `core` | `fig_krylov_comparison` |
| 7: circumference dependence | `finite` | `fig_finite_size_collapse` |
| 8: three-dimensional example | `cubic` | `fig_dipolar3d` |
| 9: coefficient growth | `core` | `fig_term_growth` |

For individual analyses:

```sh
build/analyze summary results/general.csv actual_trace_distance
build/analyze features results/ring_.04_1e-4.csv fid_average
build/analyze compare results/fid16_fine.csv F results/fulltrace16.csv F
build/analyze containment results/krylov_tpgp.csv fid_average results/lanczos.csv lower95 upper95
build/compare_tables candidate.csv reference.csv time fid_average 1e-12
```

Curve comparisons use linear interpolation only within the common time
interval and identify it as a numerical comparison. Interval-membership checks
use matching sampled times only. Richardson extrapolation assumes a factor-two
step sequence and uses the second-order formula; it is not used to construct
the moment or trajectory bounds.

## Numerical protocols

`engine/include/manuscript_inputs.hpp` stores the original finite-network
couplings and Bloch vectors as binary64-round-tripping constants. These are
inputs, not precomputed answers. `pcg_inputs.hpp` reproduces the original PCG
streams for the Ising couplings and eight independent Lanczos trace vectors.
The latter use phases in `{1,i,-1,-i}`. Native tests check the input stream.

The nearest-neighbor ring, XY chain, and cubic examples use the original local
update ordering, threshold, and output grid. Exact ring moments use GMP integer
commutator recursion. Moment envelopes and Hankel determinants use rational
arithmetic; envelope endpoints are converted outward to binary64. The general
tensor reference uses complete two-spin unitary transfer matrices, not a
splitting of noncommuting components within a pair.

For the interacting ring, `orbit_fid_rev` uses secular nearest-neighbor coupling
one and the binary64 value of 0.3 for the next-nearest neighbor. Its output
column `s` is propagation time; `t=2s` is the half-time FID readout time.
`half_lo` and `half_hi` enclose that readout. The executable stops if the
interval half-width exceeds `1e-3`. Dimensionless time is measured in inverse
nearest-neighbor angular-frequency coupling units.

The bounded recipe uses the published thresholds and a larger bucket count
from the start to limit temporary memory. Bucket grouping may change the last
rounding bits; its effects are included in the arithmetic allowance. Stored
trajectories and physical conclusions are compared at matching times.

The cost recipe fixes one CPU thread and the 31-point grid `0:1/16:1.875`.
It measures each executable from process creation through exit, excluding
compilation. Peak RSS is reported in bytes on both macOS and Linux. Runtime
and memory measurements depend on hardware, compiler, numerical libraries,
and operating system; rerun measurements are not the original Geronimo timings.
The full-trace reference is a floating-point diagonalization, whereas the
orbit calculation reports a finite-precision enclosure.

## Checkpoints

```sh
TPGP_ORBIT_BUCKETS=16 TPGP_ORBIT_SAVE=first.ck \
  build/orbit_fid_rev 1000 .03125 3e-8 20 4000000 8000000 > first.csv
TPGP_ORBIT_BUCKETS=32 TPGP_ORBIT_RESTART=first.ck TPGP_ORBIT_SAVE=continued.ck \
  build/orbit_fid_rev 1000 .03125 3e-8 32 4000000 8000000 > continued.csv
```

Use a new checkpoint output filename for a restart. The input checkpoint,
model, step, threshold, accumulated bound, and arithmetic format are checked.
SHA-256 detects incomplete or modified checkpoint data. Checkpoints are tied to
the recorded long-double format; macOS and Linux formats may differ.

## License

BSD 3-Clause; see `LICENSE`.
