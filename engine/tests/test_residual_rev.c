#include "residual_rev.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef TPGP_HAVE_MPFR
#include <mpfr.h>
#endif

static unsigned checks;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)

#ifdef TPGP_HAVE_MPFR
static int zero_generator(void) {
    tpgp_map a = {0}, b = {0};
    CHECK(tpgp_map_init(&a, 1000) == TPGP_OK);
    CHECK(tpgp_map_init(&b, 1000) == TPGP_OK);
    CHECK(tpgp_map_add_single(&a, 0, 1, 3) == TPGP_OK);
    CHECK(tpgp_map_add_single(&b, 999, 3, 4) == TPGP_OK);
    for (unsigned depth = 0; depth <= 10; ++depth) {
        double bound = -1;
        CHECK(tpgp_residual_bound(&a, &b, NULL, 0, .125, depth, &bound) == TPGP_OK);
        /* H=0: Hermite residual is 6s(1-s)(b-a); its integral is exactly 5. */
        CHECK(bound >= 5 && bound <= nextafter(5, INFINITY));
        CHECK(tpgp_residual_bound(&a, &a, NULL, 0, .125, depth, &bound) == TPGP_OK);
        CHECK(bound == 0);
    }
    tpgp_map_free(&a); tpgp_map_free(&b);
    puts("residual: exact zero-generator union-of-support result at depths 0 through 10");
    return 0;
}

static int two_site_rotation(void) {
    tpgp_map a = {0}, b = {0};
    CHECK(tpgp_map_init(&a, 2) == TPGP_OK);
    CHECK(tpgp_map_init(&b, 2) == TPGP_OK);
    CHECK(tpgp_map_add_single(&a, 0, 1, 1) == TPGP_OK);
    const uint64_t p = 1, r = 2 | (UINT64_C(3) << 2);
    mpfr_t angle, cosine, sine, x, y, error;
    mpfr_inits2(1024, angle, cosine, sine, x, y, error, (mpfr_ptr)0);
    for (int direction = -1; direction <= 1; direction += 2) {
        tpgp_term h = {0, 1, 3, 3, direction * .5, 0};
        double dt = .125;
        mpfr_set_d(angle, direction * dt, MPFR_RNDN);
        mpfr_cos(cosine, angle, MPFR_RNDN);
        mpfr_sin(sine, angle, MPFR_RNDN);
        const double bc = mpfr_get_d(cosine, MPFR_RNDN);
        const double bs = mpfr_get_d(sine, MPFR_RNDN);
        tpgp_map_free(&b);
        CHECK(tpgp_map_init(&b, 2) == TPGP_OK);
        CHECK(tpgp_map_add(&b, &p, bc) == TPGP_OK);
        CHECK(tpgp_map_add(&b, &r, bs) == TPGP_OK);
        double previous = INFINITY;
        for (unsigned depth = 0; depth <= 6; ++depth) {
            double bound;
            CHECK(tpgp_residual_bound(&a, &b, &h, 1, dt, depth, &bound) == TPGP_OK);
            CHECK(bound > 0 && bound < 1e-5);
            CHECK(bound <= previous * (1 + 4 * DBL_EPSILON));
            previous = bound;
        }
        /* Deliberately retain only the first-order endpoint. Its actual
         * error has a closed analytic expression, independent of the engine. */
        tpgp_map_free(&b);
        CHECK(tpgp_map_init(&b, 2) == TPGP_OK);
        CHECK(tpgp_map_add(&b, &p, 1) == TPGP_OK);
        CHECK(tpgp_map_add(&b, &r, direction * dt) == TPGP_OK);
        mpfr_ui_sub(x, 1, cosine, MPFR_RNDN);
        mpfr_sqr(x, x, MPFR_RNDN);
        mpfr_sub(y, angle, sine, MPFR_RNDN);
        mpfr_sqr(y, y, MPFR_RNDN);
        mpfr_add(error, x, y, MPFR_RNDN);
        mpfr_sqrt(error, error, MPFR_RNDN);
        double bound;
        CHECK(tpgp_residual_bound(&a, &b, &h, 1, dt, 5, &bound) == TPGP_OK);
        CHECK(mpfr_cmp_d(error, bound) <= 0);
        CHECK(bound < .009);
    }
    mpfr_clears(angle, cosine, sine, x, y, error, (mpfr_ptr)0);
    tpgp_map_free(&a); tpgp_map_free(&b);
    puts("residual: correct signed rotations, subdivision tightening, and analytic endpoint-error containment");
    return 0;
}

static int extreme_scales(void) {
    tpgp_map a = {0};
    CHECK(tpgp_map_init(&a, 2) == TPGP_OK);
    CHECK(tpgp_map_add_single(&a, 0, 1, 1) == TPGP_OK);
    tpgp_term huge = {0, 1, 3, 3, 0x1p1022, 0};
    tpgp_term ordinary = {0, 1, 3, 3, .5, 0};
    double large_scale, ordinary_scale;
    CHECK(tpgp_residual_bound(&a, &a, &huge, 1, 0x1p-1022, 4, &large_scale) == TPGP_OK);
    CHECK(tpgp_residual_bound(&a, &a, &ordinary, 1, 2, 4, &ordinary_scale) == TPGP_OK);
    CHECK(large_scale == ordinary_scale && large_scale > 0);
    huge.h = 0x1p1023;
    CHECK(tpgp_residual_bound(&a, &a, &huge, 1, DBL_TRUE_MIN, 4, &large_scale) == TPGP_OK);
    CHECK(tpgp_residual_bound(&a, &a, &ordinary, 1, 0x1p-50, 4, &ordinary_scale) == TPGP_OK);
    CHECK(large_scale == ordinary_scale && large_scale > 0);
    tpgp_map_free(&a);
    puts("residual: dyadic rescaling equivalence survives extreme and subnormal physical timesteps");
    return 0;
}

static int malformed_inputs(void) {
    uint64_t keys[2] = {1, 2};
    double coefficients[2] = {1, 2};
    tpgp_map valid = {2, 1, 2, 2, keys, coefficients};
    double bound;
#define REJECT(m, terms, count, dt, depth) do { \
    bound = 123; \
    CHECK(tpgp_residual_bound(m, &valid, terms, count, dt, depth, &bound) == TPGP_EINVAL); \
    CHECK(bound == 123); \
} while (0)
    tpgp_map invalid = valid;
    REJECT(NULL, NULL, 0, .1, 0);
    invalid.n_spins = 0; REJECT(&invalid, NULL, 0, .1, 0); invalid = valid;
    invalid.n_words = 0; REJECT(&invalid, NULL, 0, .1, 0); invalid = valid;
    invalid.n_words = 2; REJECT(&invalid, NULL, 0, .1, 0); invalid = valid;
    invalid.capacity = 1; REJECT(&invalid, NULL, 0, .1, 0); invalid = valid;
    invalid.capacity = SIZE_MAX; REJECT(&invalid, NULL, 0, .1, 0); invalid = valid;
    invalid.keys = NULL; REJECT(&invalid, NULL, 0, .1, 0); invalid = valid;
    invalid.values = NULL; REJECT(&invalid, NULL, 0, .1, 0); invalid = valid;
    invalid.keys = NULL; invalid.len = 0; REJECT(&invalid, NULL, 0, .1, 0); invalid = valid;
    keys[1] = keys[0]; REJECT(&valid, NULL, 0, .1, 0); keys[1] = 2;
    keys[0] = 3; REJECT(&valid, NULL, 0, .1, 0); keys[0] = 1;
    keys[1] = 16; REJECT(&valid, NULL, 0, .1, 0); keys[1] = 2;
    coefficients[0] = NAN; REJECT(&valid, NULL, 0, .1, 0); coefficients[0] = 1;
    coefficients[0] = INFINITY; REJECT(&valid, NULL, 0, .1, 0); coefficients[0] = 1;
    REJECT(&valid, NULL, 1, .1, 0);
    REJECT(&valid, NULL, 0, .1, 11);
    REJECT(&valid, NULL, 0, 0, 0);
    REJECT(&valid, NULL, 0, NAN, 0);
    tpgp_term term = {0, 0, 1, 1, 1, 0};
    REJECT(&valid, &term, 1, .1, 0);
    term.j = 1; term.qi = 4; REJECT(&valid, &term, 1, .1, 0);
    term.qi = 1; term.h = INFINITY; REJECT(&valid, &term, 1, .1, 0);
    term.h = 1; REJECT(&valid, &term, SIZE_MAX, .1, 0);
#undef REJECT
    uint64_t p = 1;
    double ca = DBL_MAX, cb = -DBL_MAX;
    tpgp_map a = {2, 1, 1, 1, &p, &ca}, b = {2, 1, 1, 1, &p, &cb};
    bound = 123;
    CHECK(tpgp_residual_bound(&a, &b, NULL, 0, 1, 0, &bound) == TPGP_ENUMERIC);
    CHECK(bound == 123);
    mpfr_exp_t old_emax = mpfr_get_emax();
    CHECK(mpfr_set_emax(100) == 0);
    CHECK(tpgp_residual_bound(&valid, &valid, NULL, 0, 1, 0, &bound) == TPGP_ENUMERIC);
    CHECK(bound == 123);
    CHECK(isnan(tpgp_bound_add_up(1, 1)));
    CHECK(isnan(tpgp_bound_fid_up(1, 4)));
    CHECK(mpfr_set_emax(old_emax) == 0);
    puts("residual: malformed maps/terms rejected; output preserved on input and numeric failures");
    return 0;
}

static int bound_helpers(void) {
    CHECK(tpgp_bound_add_up(0, 0) == 0);
    CHECK(tpgp_bound_add_up(1, 0x1p-54) == nextafter(1, INFINITY));
    CHECK(isinf(tpgp_bound_add_up(DBL_MAX, DBL_MAX)));
    CHECK(isnan(tpgp_bound_add_up(-1, 1)));
    CHECK(isnan(tpgp_bound_add_up(NAN, 1)));
    CHECK(isnan(tpgp_bound_add_up(1, -INFINITY)));
    CHECK(isinf(tpgp_bound_add_up(INFINITY, 1)));
    CHECK(isnan(tpgp_bound_fid_up(1, 0)));
    CHECK(isnan(tpgp_bound_fid_up(-1, 4)));
    CHECK(isnan(tpgp_bound_fid_up(NAN, 4)));
    CHECK(isinf(tpgp_bound_fid_up(INFINITY, 4)));
    CHECK(tpgp_bound_fid_up(3, 9) == 1);
    mpfr_t x;
    mpfr_init2(x, 1024);
    mpfr_set_ui(x, 3, MPFR_RNDN);
    mpfr_sqrt(x, x, MPFR_RNDD);
    mpfr_ui_div(x, 1, x, MPFR_RNDU);
    CHECK(mpfr_cmp_d(x, tpgp_bound_fid_up(1, 3)) <= 0);
    mpfr_clear(x);
    puts("residual: upward sum/normalization and invalid helper inputs");
    return 0;
}

static uint64_t rng_state = UINT64_C(0x1ef39af7a213b581);
static uint64_t random_word(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static int gram_reference_agreement(void) {
    for (unsigned trial = 0; trial < 12; ++trial) {
        tpgp_map a = {0}, b = {0};
        CHECK(tpgp_map_init(&a, 3) == TPGP_OK);
        CHECK(tpgp_map_init(&b, 3) == TPGP_OK);
        for (uint64_t key = 0; key < 64; ++key) {
            const double ca = ((int)(random_word() % 33) - 16) / 16.0;
            const double cb = ((int)(random_word() % 33) - 16) / 16.0;
            if (ca) CHECK(tpgp_map_add(&a, &key, ca) == TPGP_OK);
            if (cb) CHECK(tpgp_map_add(&b, &key, cb) == TPGP_OK);
        }
        tpgp_term terms[] = {{0, 1, 3, 3, .5, 0}, {1, 2, 1, 1, -.25, 0},
                             {0, 2, 2, 2, .125, 0}, {0, 1, 1, 2, -.375, 0}};
        for (unsigned depth = 0; depth <= 5; ++depth) {
            double bound, reference;
            const double dt = ldexp(1, (int)(trial % 8) - 5);
            CHECK(tpgp_residual_bound(&a, &b, terms, 4, dt, depth, &bound) == TPGP_OK);
            CHECK(tpgp_residual_bound_reference(&a, &b, terms, 4, dt, depth, &reference) == TPGP_OK);
            CHECK(bound > 0 && reference > 0);
            CHECK(fabs(bound - reference) <= 4 * DBL_EPSILON * reference);
        }
        tpgp_map_free(&a); tpgp_map_free(&b);
    }
    /* At large dt, D1 and D2 have opposite quadratic contributions. The
     * equal-weight midpoint requires cancellation of more than 256 bits;
     * direct support evaluation must remain available for that control. */
    tpgp_map a = {0};
    CHECK(tpgp_map_init(&a, 2) == TPGP_OK);
    CHECK(tpgp_map_add_single(&a, 0, 1, 1) == TPGP_OK);
    tpgp_term h = {0, 1, 3, 3, .5, 0};
    for (unsigned depth = 1; depth <= 6; ++depth) {
        double bound, reference;
        CHECK(tpgp_residual_bound(&a, &a, &h, 1, 0x1p500, depth, &bound) == TPGP_OK);
        CHECK(tpgp_residual_bound_reference(&a, &a, &h, 1, 0x1p500, depth, &reference) == TPGP_OK);
        CHECK(bound > 0 && reference > 0);
        CHECK(fabs(bound - reference) <= 4 * DBL_EPSILON * reference);
    }
    tpgp_map_free(&a);
    puts("residual: signed Gram/reference agreement for random maps and severe cancellation");
    return 0;
}

static double wall_seconds(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int benchmark(void) {
    const size_t n = 6;
    tpgp_map a = {0}, b = {0};
    CHECK(tpgp_map_init(&a, n) == TPGP_OK);
    for (size_t i = 0; i < n; ++i) CHECK(tpgp_map_add_single(&a, i, 1, 1) == TPGP_OK);
    tpgp_term terms[36];
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) for (size_t r = 1; r <= 2; ++r) {
        const double d = r == 1 ? 1 : .3;
        terms[count++] = (tpgp_term){i, (i+r)%n, 3, 3, d/2, 0};
        terms[count++] = (tpgp_term){i, (i+r)%n, 1, 1, -d/4, 0};
        terms[count++] = (tpgp_term){i, (i+r)%n, 2, 2, -d/4, 0};
    }
    tpgp_options options = tpgp_default_options();
    options.threshold = 1e-5; options.validated = 1; options.max_terms = 100000;
    tpgp_ledger ledger; tpgp_ledger_init(&ledger, 1);
    for (unsigned step = 0; step < 3; ++step)
        CHECK(tpgp_strang_step(&a, terms, count, 1.0/64, &options, &ledger) == TPGP_OK);
    CHECK(tpgp_map_clone(&b, &a) == TPGP_OK);
    CHECK(tpgp_strang_step(&b, terms, count, 1.0/64, &options, &ledger) == TPGP_OK);
    const unsigned depths[] = {0, 1, 4, 7};
    for (size_t k = 0; k < sizeof depths / sizeof depths[0]; ++k) {
        double optimized, reference;
        const double start = wall_seconds();
        CHECK(tpgp_residual_bound_reference(&a, &b, terms, count, 1.0/64,
                                          depths[k], &reference) == TPGP_OK);
        const double middle = wall_seconds();
        CHECK(tpgp_residual_bound(&a, &b, terms, count, 1.0/64,
                                depths[k], &optimized) == TPGP_OK);
        const double finish = wall_seconds();
        CHECK(fabs(optimized - reference) <= 4 * DBL_EPSILON * reference);
        printf("residual_benchmark,N=%zu,active_a=%zu,active_b=%zu,depth=%u,"
               "reference_seconds=%.6f,gram_seconds=%.6f,reference_bound=%.17g,gram_bound=%.17g\n",
               n, a.len, b.len, depths[k], middle-start, finish-middle, reference, optimized);
    }
    tpgp_map_free(&a); tpgp_map_free(&b);
    return 0;
}
#endif

int main(int argc, char **argv) {
#ifndef TPGP_HAVE_MPFR
    (void)argc; (void)argv;
    double untouched = 123;
    CHECK(tpgp_residual_bound(NULL, NULL, NULL, 0, 1, 0, &untouched) == TPGP_EBACKEND);
    CHECK(untouched == 123);
    CHECK(isnan(tpgp_bound_fid_up(1, 1)));
    puts("residual: MPFR absent; capability failure is explicit (enclosure tests not run)");
    return 0;
#else
    if (argc == 2 && !strcmp(argv[1], "--benchmark")) return benchmark();
    if (zero_generator() || two_site_rotation() || extreme_scales() ||
        malformed_inputs() || bound_helpers() || gram_reference_agreement()) return EXIT_FAILURE;
    printf("residual: %u checks passed\n", checks);
    return EXIT_SUCCESS;
#endif
}
