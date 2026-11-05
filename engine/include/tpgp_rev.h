#ifndef TPGP_REV_H
#define TPGP_REV_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TPGP_OK = 0,
    TPGP_EINVAL,
    TPGP_ENOMEM,
    TPGP_ECAP,
    TPGP_ENUMERIC,
    TPGP_EBACKEND,
    TPGP_EBUDGET
} tpgp_status;
typedef enum {
    TPGP_SERIAL = 0,
    TPGP_PTHREAD = 1,
    TPGP_OPENMP = 2,
    TPGP_METAL = 3,
    TPGP_CUDA = 4
} tpgp_backend;
typedef struct {
    size_t n_spins, n_words, len, capacity;
    uint64_t *keys; /* keys[k*n_words+w]; two bits per site */
    double *values; /* A=sum_k values[k] P_k */
} tpgp_map;
typedef struct {
    size_t i, j;
    unsigned qi, qj;
    double h;     /* Hamiltonian contribution h Q */
    size_t layer; /* optional commuting-block label; array order is authoritative */
} tpgp_term;
typedef struct {
    double threshold; /* remove merged |b| <= threshold */
    size_t max_terms; /* 0 means no imposed active-term cap */
    unsigned threads;
    tpgp_backend backend;
    int validated;       /* MPFR trig enclosures + IEEE binary64 bounds */
    double max_drop_l2;  /* 0 means no imposed cumulative drop budget */
    void *metal_context; /* optional reusable tpgp_metal_context, caller-owned */
    void *cuda_context;  /* optional reusable tpgp_cuda_context, caller-owned */
} tpgp_options;
typedef struct {
    double dropped_l1, dropped_l2, dropped_l2_squared;
    double arithmetic_l2;
    size_t rotations, removal_events, peak_terms;
    int arithmetic_enclosed;
} tpgp_ledger;

const char *tpgp_status_string(tpgp_status status);
int tpgp_has_mpfr(void);
int tpgp_has_openmp(void);
tpgp_options tpgp_default_options(void);
void tpgp_ledger_init(tpgp_ledger *ledger, int validated);
/* Maps must initially be zero-initialized; do not mutate owned storage or
   structural fields directly. init requires an empty map; free resets it. */
tpgp_status tpgp_map_init(tpgp_map *map, size_t n_spins);
void tpgp_map_free(tpgp_map *map);
tpgp_status tpgp_map_clone(tpgp_map *dst, const tpgp_map *src);
unsigned tpgp_key_get(const uint64_t *key, size_t site);
void tpgp_key_set(uint64_t *key, size_t site, unsigned pauli);
/* Add a finite coefficient; maps remain lexicographically sorted. */
tpgp_status tpgp_map_add(tpgp_map *map, const uint64_t *key, double value);
tpgp_status tpgp_map_add_single(tpgp_map *map, size_t site, unsigned pauli, double value);
double tpgp_map_get(const tpgp_map *map, const uint64_t *key);
double tpgp_map_l1(const tpgp_map *map);
double tpgp_map_l2(const tpgp_map *map);
double tpgp_map_distance(const tpgp_map *a, const tpgp_map *b);
double tpgp_map_dot(const tpgp_map *a, const tpgp_map *b);
double tpgp_map_single_sum(const tpgp_map *map, unsigned pauli);
/* Correctly enclosed sum of single-Pauli coefficients / normalizer.
   midpoint is binary64; radius encloses its readout-rounding error.
   Requires MPFR, otherwise returns EBACKEND without changing outputs. */
tpgp_status tpgp_readout(const tpgp_map *map, unsigned pauli, size_t normalizer, double *midpoint,
                         double *radius);
/* dst <- a + scale*b; dst may alias either input. */
tpgp_status tpgp_map_add_scaled(tpgp_map *dst, const tpgp_map *a, const tpgp_map *b, double scale);
/* phase is the exponent of i in QP; result local symbols are q XOR p. */
unsigned tpgp_pauli_phase(unsigned q, unsigned p);
/* Every mutation below is transactional: map and ledger unchanged on failure. */
tpgp_status tpgp_rotate(tpgp_map *map, size_t i, size_t j, unsigned qi, unsigned qj, double theta,
                        const tpgp_options *options, tpgp_ledger *ledger);
tpgp_status tpgp_strang_step(tpgp_map *map, const tpgp_term *terms, size_t count, double duration,
                             const tpgp_options *options, tpgp_ledger *ledger);
/* Exact algebraic generator -i[H,A], evaluated in binary64; no pruning. */
tpgp_status tpgp_liouvillian(tpgp_map *out, const tpgp_map *in, const tpgp_term *terms,
                             size_t count);
/* Conservative third-order Strang bound for ordered individual terms:
   ||(exp(dt L)-S(dt)) A||_2 <= bound * ||A||_2.
   This is a global operator-norm bound, not an observed error. */
double tpgp_strang_bound(const tpgp_term *terms, size_t count, double duration);
/* Allocation-free Taylor fallback: (2/3)(|dt| sum_j |h_j|)^3. */
double tpgp_strang_bound_taylor(const tpgp_term *terms, size_t count, double duration);

#ifdef __cplusplus
}
#endif
#endif
