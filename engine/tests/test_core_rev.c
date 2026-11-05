#include "tpgp_metal_rev.h"
#include "tpgp_rev.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef TPGP_HAVE_MPFR
#include <mpfr.h>
#endif

static int failures = 0, checks = 0;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        checks++;                                                                                  \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                           \
            failures++;                                                                            \
        }                                                                                          \
    } while (0)
static void pauli(unsigned a, unsigned b, double complex out[16]) {
    static const double complex p[4][4] = {
        {1, 0, 0, 1}, {0, 1, 1, 0}, {0, -I, I, 0}, {1, 0, 0, -1}};
    for (unsigned r = 0; r < 4; r++)
        for (unsigned c = 0; c < 4; c++)
            out[4 * r + c] = p[a][2 * (r & 1) + (c & 1)] * p[b][2 * (r >> 1) + (c >> 1)];
}
static void mul(const double complex a[16], const double complex b[16], double complex c[16]) {
    for (unsigned i = 0; i < 4; i++)
        for (unsigned j = 0; j < 4; j++) {
            c[4 * i + j] = 0;
            for (unsigned k = 0; k < 4; k++)
                c[4 * i + j] += a[4 * i + k] * b[4 * k + j];
        }
}
static void local_dense(void) {
    tpgp_options o = tpgp_default_options();
    o.validated = tpgp_has_mpfr();
    double theta = .371;
    for (unsigned qa = 0; qa < 4; qa++)
        for (unsigned qb = 0; qb < 4; qb++)
            for (unsigned pa = 0; pa < 4; pa++)
                for (unsigned pb = 0; pb < 4; pb++) {
                    double complex q[16], p[16], u[16], ud[16], tmp[16], result[16];
                    pauli(qa, qb, q);
                    pauli(pa, pb, p);
                    for (unsigned r = 0; r < 4; r++)
                        for (unsigned c = 0; c < 4; c++) {
                            u[4 * r + c] =
                                (r == c ? cos(theta) : 0) - I * sin(theta) * q[4 * r + c];
                            ud[4 * r + c] =
                                (r == c ? cos(theta) : 0) + I * sin(theta) * q[4 * r + c];
                        }
                    mul(u, p, tmp);
                    mul(tmp, ud, result);
                    tpgp_map m = {0};
                    CHECK(tpgp_map_init(&m, 2) == TPGP_OK);
                    uint64_t key = (uint64_t)pa | ((uint64_t)pb << 2);
                    CHECK(tpgp_map_add(&m, &key, 1) == TPGP_OK);
                    tpgp_ledger l;
                    tpgp_ledger_init(&l, o.validated);
                    CHECK(tpgp_rotate(&m, 0, 1, qa, qb, theta, &o, &l) == TPGP_OK);
                    for (unsigned a = 0; a < 4; a++)
                        for (unsigned b = 0; b < 4; b++) {
                            double complex basis[16];
                            pauli(a, b, basis);
                            double complex coefficient = 0;
                            for (unsigned r = 0; r < 4; r++)
                                for (unsigned c = 0; c < 4; c++)
                                    coefficient += conj(basis[4 * r + c]) * result[4 * r + c] / 4;
                            uint64_t k = a | ((uint64_t)b << 2);
                            CHECK(fabs(tpgp_map_get(&m, &k) - creal(coefficient)) < 8e-16);
                            CHECK(fabs(cimag(coefficient)) < 1e-15);
                        }
                    CHECK(fabs(tpgp_map_l2(&m) - 1) < 5e-16);
                    tpgp_map_free(&m);
                }
    printf("local_dense: all 256 Pauli conjugations agree with independent 4x4 complex matrices\n");
}
static void rollback_and_ledger(void) {
    tpgp_map m = {0}, original = {0};
    tpgp_map_init(&m, 2);
    tpgp_map_add_single(&m, 0, 1, 1);
    tpgp_map_clone(&original, &m);
    tpgp_options o = tpgp_default_options();
    o.validated = tpgp_has_mpfr();
    o.max_terms = 1;
    tpgp_ledger l, old;
    tpgp_ledger_init(&l, o.validated);
    old = l;
    CHECK(tpgp_rotate(&m, 0, 1, 3, 3, .2, &o, &l) == TPGP_ECAP);
    CHECK(tpgp_map_distance(&m, &original) == 0);
    CHECK(memcmp(&old, &l, sizeof(l)) == 0);
    o.max_terms = 0;
    o.threshold = .03;
    CHECK(tpgp_rotate(&m, 0, 1, 3, 3, .01, &o, &l) == TPGP_OK);
    CHECK(tpgp_rotate(&m, 0, 1, 3, 3, .01, &o, &l) == TPGP_OK);
    CHECK(m.len == 1);
    CHECK(l.removal_events == 2);
    CHECK(l.dropped_l2 >= .03998);
    CHECK(l.dropped_l2 > sqrt(l.dropped_l2_squared) * 1.4);
    CHECK(l.dropped_l2 >= l.dropped_l1 && l.dropped_l2 - l.dropped_l1 < 1e-15);
    tpgp_map_clone(&original, &m);
    old = l;
    o.max_drop_l2 = l.dropped_l2;
    CHECK(tpgp_rotate(&m, 0, 1, 3, 3, .01, &o, &l) == TPGP_EBUDGET);
    CHECK(tpgp_map_distance(&m, &original) == 0);
    CHECK(memcmp(&old, &l, sizeof(l)) == 0);
    tpgp_map_free(&m);
    tpgp_map_free(&original);
    printf("ledger_and_resource: eventwise L2 sums and transactional cap/budget checks passed\n");
}
static void arbitrary_n_parallel(void) {
    tpgp_map m = {0}, a = {0}, b = {0};
    CHECK(tpgp_map_init(&m, 1000) == TPGP_OK);
    for (size_t k = 0; k < 100; k++)
        CHECK(tpgp_map_add_single(&m, k * 10, 1, 1.0 / (k + 1)) == TPGP_OK);
    CHECK(tpgp_map_add_single(&m, 999, 2, .6) == TPGP_OK);
    CHECK(tpgp_map_clone(&a, &m) == TPGP_OK);
    CHECK(tpgp_map_clone(&b, &m) == TPGP_OK);
    tpgp_options o = tpgp_default_options();
    o.validated = tpgp_has_mpfr();
    tpgp_ledger la, lb;
    tpgp_ledger_init(&la, o.validated);
    tpgp_ledger_init(&lb, o.validated);
    CHECK(tpgp_rotate(&a, 0, 999, 3, 3, .203, &o, &la) == TPGP_OK);
    o.backend = TPGP_PTHREAD;
    o.threads = 4;
    CHECK(tpgp_rotate(&b, 0, 999, 3, 3, .203, &o, &lb) == TPGP_OK);
    CHECK(a.len == b.len);
    CHECK(memcmp(a.keys, b.keys, a.len * a.n_words * sizeof(uint64_t)) == 0);
    CHECK(memcmp(a.values, b.values, a.len * sizeof(double)) == 0);
    CHECK(la.arithmetic_l2 == lb.arithmetic_l2);
    if (tpgp_has_openmp()) {
        CHECK(tpgp_map_clone(&b, &m) == TPGP_OK);
        o.backend = TPGP_OPENMP;
        tpgp_ledger_init(&lb, o.validated);
        CHECK(tpgp_rotate(&b, 0, 999, 3, 3, .203, &o, &lb) == TPGP_OK);
        CHECK(a.len == b.len);
        CHECK(memcmp(a.values, b.values, a.len * sizeof(double)) == 0);
        CHECK(memcmp(a.keys, b.keys, a.len * a.n_words * sizeof(uint64_t)) == 0);
    }
    o.backend = TPGP_SERIAL;
    CHECK(tpgp_rotate(&a, 0, 999, 3, 3, -.203, &o, &la) == TPGP_OK);
    double reverse = tpgp_map_distance(&a, &m);
    CHECK(reverse < 4e-16);
    if (o.validated)
        CHECK(reverse <= la.arithmetic_l2);
    CHECK(fabs(tpgp_map_l2(&a) - tpgp_map_l2(&m)) < 5e-16);
    uint64_t *key = calloc(m.n_words, sizeof(uint64_t));
    tpgp_key_set(key, 999, 3);
    CHECK(tpgp_key_get(key, 999) == 3);
    CHECK(key[31] == (UINT64_C(3) << 14));
    free(key);
    tpgp_map_free(&m);
    tpgp_map_free(&a);
    tpgp_map_free(&b);
    printf("N1000: high-bit keys, deterministic pthread/OpenMP, norm and reversal passed (distance "
           "%.3g)\n",
           reverse);
}
static void generator_and_strang(void) {
    tpgp_map m = {0}, g = {0}, a = {0}, fine = {0}, comb = {0};
    tpgp_map_init(&m, 3);
    tpgp_map_add_single(&m, 0, 1, 1);
    tpgp_term terms[] = {{0, 1, 3, 3, .5, 0}, {1, 2, 1, 1, -.25, 1}, {0, 2, 2, 2, .3, 2}};
    CHECK(tpgp_liouvillian(&g, &m, terms, 3) == TPGP_OK);
    uint64_t yz = 2 | (UINT64_C(3) << 2), zy = 3 | (UINT64_C(2) << 4);
    CHECK(tpgp_map_get(&g, &yz) == 1);
    CHECK(fabs(tpgp_map_get(&g, &zy) + .6) < 1e-16);
    CHECK(fabs(tpgp_map_dot(&m, &g)) < 1e-16);
    CHECK(tpgp_map_add_scaled(&comb, &m, &g, .2) == TPGP_OK);
    CHECK(comb.len == 3);
    tpgp_options o = tpgp_default_options();
    o.validated = tpgp_has_mpfr();
    tpgp_ledger l;
    tpgp_ledger_init(&l, o.validated);
    tpgp_map_clone(&a, &m);
    tpgp_map_clone(&fine, &m);
    CHECK(tpgp_strang_step(&a, terms, 3, .04, &o, &l) == TPGP_OK);
    for (int i = 0; i < 40; i++)
        CHECK(tpgp_strang_step(&fine, terms, 3, .001, &o, &l) == TPGP_OK);
    double error = tpgp_map_distance(&a, &fine);
    CHECK(error < tpgp_strang_bound(terms, 3, .04));
    CHECK(error > 1e-9);
    CHECK(fabs(tpgp_map_l2(&a) - 1) < 8e-16);
    tpgp_map_clone(&comb, &a);
    tpgp_ledger old = l;
    o.max_terms = 1;
    CHECK(tpgp_strang_step(&a, terms, 3, .04, &o, &l) == TPGP_ECAP);
    CHECK(tpgp_map_distance(&a, &comb) == 0);
    CHECK(memcmp(&l, &old, sizeof(l)) == 0);
    tpgp_map_free(&m);
    tpgp_map_free(&g);
    tpgp_map_free(&a);
    tpgp_map_free(&fine);
    tpgp_map_free(&comb);
    printf("generator_and_strang: algebra, norm, small-step reference, whole-step rollback passed "
           "(distance %.3g)\n",
           error);
}
static void backend_trajectories(void) {
    tpgp_map original = {0}, reference = {0};
    tpgp_map_init(&original, 4);
    for (size_t s = 0; s < 4; s++)
        tpgp_map_add_single(&original, s, 1, 1);
    tpgp_map_clone(&reference, &original);
    tpgp_term terms[12];
    for (size_t k = 0; k < 4; k++) {
        terms[3 * k] = (tpgp_term){k, (k + 1) % 4, 3, 3, .5, k};
        terms[3 * k + 1] = (tpgp_term){k, (k + 1) % 4, 1, 1, -.25, k};
        terms[3 * k + 2] = (tpgp_term){k, (k + 1) % 4, 2, 2, -.25, k};
    }
    tpgp_options o = tpgp_default_options();
    o.validated = tpgp_has_mpfr();
    o.threshold = 1e-9;
    tpgp_ledger expected;
    tpgp_ledger_init(&expected, o.validated);
    for (int k = 0; k < 4; k++)
        CHECK(tpgp_strang_step(&reference, terms, 12, .02, &o, &expected) == TPGP_OK);
    for (int backend = TPGP_PTHREAD; backend <= TPGP_METAL; backend++) {
        if (backend == TPGP_OPENMP && !tpgp_has_openmp())
            continue;
        char err[256];
        if (backend == TPGP_METAL) {
            o.metal_context = tpgp_metal_create(err, sizeof(err));
            if (!o.metal_context) {
                printf("whole_trajectory_metal: unavailable (%s); no CPU substitution\n", err);
                continue;
            }
        }
        tpgp_map test = {0};
        tpgp_map_clone(&test, &original);
        tpgp_ledger actual;
        tpgp_ledger_init(&actual, o.validated);
        o.backend = (tpgp_backend)backend;
        o.threads = 4;
        for (int k = 0; k < 4; k++)
            CHECK(tpgp_strang_step(&test, terms, 12, .02, &o, &actual) == TPGP_OK);
        CHECK(reference.len == test.len);
        CHECK(reference.len != test.len ||
              memcmp(reference.keys, test.keys, test.len * test.n_words * sizeof(uint64_t)) == 0);
        CHECK(reference.len != test.len ||
              memcmp(reference.values, test.values, test.len * sizeof(double)) == 0);
        CHECK(expected.dropped_l1 == actual.dropped_l1);
        CHECK(expected.dropped_l2 == actual.dropped_l2);
        CHECK(expected.arithmetic_l2 == actual.arithmetic_l2);
        CHECK(expected.rotations == actual.rotations);
        printf("whole_trajectory_backend=%d: exact binary64 agreement with serial at all final "
               "coefficients and ledgers\n",
               backend);
        tpgp_map_free(&test);
        if (o.metal_context) {
            tpgp_metal_destroy(o.metal_context);
            o.metal_context = NULL;
        }
    }
    if (tpgp_has_mpfr()) {
        double value = 0, radius = 0;
        CHECK(tpgp_readout(&reference, 1, 4, &value, &radius) == TPGP_OK);
        CHECK(radius >= 0);
        CHECK(fabs(value - tpgp_map_single_sum(&reference, 1) / 4) <= radius + 2e-16);
        tpgp_map single = {0};
        tpgp_map_init(&single, 2);
        tpgp_map_add_single(&single, 0, 1, 1);
        CHECK(tpgp_readout(&single, 1, 3, &value, &radius) == TPGP_OK);
        CHECK(value == 1.0 / 3.0);
        CHECK(radius > 0 && radius < 2e-17);
        tpgp_map_free(&single);
    }
    tpgp_map_free(&original);
    tpgp_map_free(&reference);
}
static void bound_fallback(void) {
    tpgp_term terms[] = {{0, 1, 3, 3, .5, 0}, {1, 2, 1, 1, .25, 1}, {0, 2, 2, 2, .25, 2}};
    double fallback = tpgp_strang_bound_taylor(terms, 3, .125);
    CHECK(fallback >= 1.0 / 768.0);
    CHECK(fallback < (1.0 / 768.0) * (1 + 3e-15));
    CHECK(tpgp_strang_bound(terms, 3, .125) <= fallback);
    CHECK(tpgp_strang_bound_taylor(NULL, 0, 1) == 0);
    CHECK(tpgp_strang_bound_taylor(terms, 3, 0) == 0);
    CHECK(tpgp_strang_bound_taylor(terms, 3, -.125) == fallback);
    printf("Taylor_fallback: Hamiltonian/conjugation bound has the 2/3 prefactor\n");
}
static void analytic_ising_N1000(void) {
    const size_t n = 1000;
    const double time = .125;
    tpgp_map map = {0};
    CHECK(tpgp_map_init(&map, n) == TPGP_OK);
    tpgp_term *terms = calloc(n, sizeof(*terms));
    CHECK(terms != NULL);
    if (!terms) {
        tpgp_map_free(&map);
        return;
    }
    for (size_t k = 0; k < n; k++) {
        CHECK(tpgp_map_add_single(&map, k, 1, 1) == TPGP_OK);
        terms[k] = (tpgp_term){k, (k + 1) % n, 3, 3, .25, 0};
    }
    tpgp_options options = tpgp_default_options();
    options.validated = tpgp_has_mpfr();
    options.max_terms = 4000;
    tpgp_ledger ledger;
    tpgp_ledger_init(&ledger, options.validated);
    CHECK(tpgp_strang_bound(terms, n, time) == 0);
    CHECK(tpgp_strang_step(&map, terms, n, time, &options, &ledger) == TPGP_OK);
    CHECK(map.len == 4000);
    double value = tpgp_map_single_sum(&map, 1) / n;
    double error = fabs(value - cos(time / 2) * cos(time / 2));
    CHECK(error < 2e-14);
#ifdef TPGP_HAVE_MPFR
    double radius = 0;
    CHECK(tpgp_readout(&map, 1, n, &value, &radius) == TPGP_OK);
    mpfr_t lo, hi, x, diff, other, bound, den;
    mpfr_inits2(256, lo, hi, x, diff, other, bound, den, (mpfr_ptr)0);
    mpfr_set_d(x, time / 2, MPFR_RNDN);
    mpfr_cos(lo, x, MPFR_RNDD);
    mpfr_sqr(lo, lo, MPFR_RNDD);
    mpfr_cos(hi, x, MPFR_RNDU);
    mpfr_sqr(hi, hi, MPFR_RNDU);
    mpfr_set_d(x, value, MPFR_RNDN);
    mpfr_sub(diff, x, lo, MPFR_RNDU);
    mpfr_sub(other, hi, x, MPFR_RNDU);
    mpfr_max(diff, diff, other, MPFR_RNDU);
    mpfr_set_ui(den, n, MPFR_RNDN);
    mpfr_sqrt(den, den, MPFR_RNDD);
    mpfr_set_d(bound, ledger.arithmetic_l2, MPFR_RNDN);
    mpfr_div(bound, bound, den, MPFR_RNDU);
    mpfr_add_d(bound, bound, radius, MPFR_RNDU);
    CHECK(mpfr_cmp(diff, bound) <= 0);
    printf("Ising_N1000: F=%.17g; analytic error <= %.4g, enclosed error bound %.4g\n", value,
           mpfr_get_d(diff, MPFR_RNDU), mpfr_get_d(bound, MPFR_RNDU));
    mpfr_clears(lo, hi, x, diff, other, bound, den, (mpfr_ptr)0);
#else
    printf("Ising_N1000: F=%.17g; binary64 analytic comparison error %.4g (no enclosure)\n", value,
           error);
#endif
    CHECK(ledger.dropped_l1 == 0 && ledger.dropped_l2 == 0);
    tpgp_map_free(&map);
    free(terms);
}
int main(void) {
    local_dense();
    rollback_and_ledger();
    arbitrary_n_parallel();
    generator_and_strang();
    backend_trajectories();
    bound_fallback();
    analytic_ising_N1000();
    printf("checks=%d failures=%d MPFR=%d OpenMP=%d\n", checks, failures, tpgp_has_mpfr(),
           tpgp_has_openmp());
    return failures ? 1 : 0;
}
