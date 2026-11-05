#ifndef RESIDUAL_REV_H
#define RESIDUAL_REV_H
#include "tpgp_rev.h"
/* Bound the normalized-Hilbert-Schmidt endpoint error generated over one
   physical-time interval by the cubic Hermite path through a and b.
   The endpoints, dt, and Hamiltonian doubles are interpreted as exact dyadics.
   All verifier arithmetic is enclosed with directed MPFR rounding.
   This bounds combined propagation errors; do NOT add the component ledger.
   Maps must have consistent dimensions/capacity, finite coefficients, strictly
   ordered unique keys (highest word first), and zero unused key bits, as maps
   constructed by the public map API do. Caller-owned arrays must actually
   contain the stated capacity. Invalid maps are rejected before propagation.
   Return EBACKEND if MPFR was not compiled in; EINVAL for malformed input;
   ENOMEM for allocation failure; ENUMERIC for an unrepresentable finite-double
   upper bound or unsupported arithmetic environment. Requires IEEE binary64,
   size_t of at most 64 bits, and MPFR exponent range covering [-16384,16384]
   (the default MPFR range is much wider). The output is unchanged on failure.
   depth <= 10. */
tpgp_status tpgp_residual_bound(const tpgp_map *a,const tpgp_map *b,
    const tpgp_term *terms,size_t count,double dt,unsigned depth,double *upper);
/* Diagnostic implementation with identical mathematical contract. It scans
   sparse supports separately for each subdivision control instead of using
   the enclosed 2-by-2 Gram matrix. Both are upper bounds; the last rounded bit
   need not agree. The default falls back to this norm evaluation in severely
   cancelling Gram combinations and uses it directly at depth zero. */
tpgp_status tpgp_residual_bound_reference(const tpgp_map *a,const tpgp_map *b,
    const tpgp_term *terms,size_t count,double dt,unsigned depth,double *upper);
/* Outward-rounded accumulated sum and normalization by sqrt(N).
   Negative/NaN inputs (or N=0) return NaN. Positive infinity is retained as
   an uninformative but valid upper bound. An unsupported arithmetic environment
   returns NaN. fid_up requires MPFR; its sqrt(N) normalization is for global
   transverse magnetization with normalized-Hilbert-Schmidt norm sqrt(N). */
double tpgp_bound_add_up(double a,double b);
double tpgp_bound_fid_up(double raw_bound,size_t n_spins);
#endif
