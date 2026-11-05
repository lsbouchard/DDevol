#include "residual_rev.h"
#include "tpgp_cuda_rev.h"
#include "tpgp_rev.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures, checks;
#define CHECK(x) do { checks++; if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); failures++; } } while (0)
static int same_double(double a, double b) { return !memcmp(&a, &b, sizeof(double)); }
static void compare(const tpgp_map *a, const tpgp_map *b,
                    const tpgp_ledger *x, const tpgp_ledger *y) {
    CHECK(a->len == b->len);
    if (a->len == b->len) {
        CHECK(!memcmp(a->keys, b->keys, a->len * a->n_words * sizeof(uint64_t)));
        CHECK(!memcmp(a->values, b->values, a->len * sizeof(double)));
    }
    CHECK(same_double(x->dropped_l1, y->dropped_l1));
    CHECK(same_double(x->dropped_l2, y->dropped_l2));
    CHECK(same_double(x->dropped_l2_squared, y->dropped_l2_squared));
    CHECK(same_double(x->arithmetic_l2, y->arithmetic_l2));
    CHECK(x->rotations == y->rotations && x->removal_events == y->removal_events &&
          x->peak_terms == y->peak_terms && x->arithmetic_enclosed == y->arithmetic_enclosed);
}
static void trajectory(tpgp_cuda_context *context, double threshold, int validated) {
    enum { N = 6, NT = 6*N };
    tpgp_map cpu = {0}, gpu = {0};
    CHECK(tpgp_map_init(&cpu, N) == TPGP_OK);
    for (size_t k = 0; k < N; k++)
        CHECK(tpgp_map_add_single(&cpu, k, 1, 1) == TPGP_OK);
    CHECK(tpgp_map_clone(&gpu, &cpu) == TPGP_OK);
    tpgp_term terms[NT];
    size_t count = 0;
    for (size_t k = 0; k < N; k++)
        for (size_t range = 1; range <= 2; range++) {
            double d = range == 1 ? 1 : .3;
            terms[count] = (tpgp_term){k, (k+range)%N, 3, 3, d/2, count}; count++;
            terms[count] = (tpgp_term){k, (k+range)%N, 1, 1, -d/4, count}; count++;
            terms[count] = (tpgp_term){k, (k+range)%N, 2, 2, -d/4, count}; count++;
        }
    tpgp_options a = tpgp_default_options(), b;
    a.threshold = threshold; a.validated = validated; a.max_terms = 100000;
    b = a; b.backend = TPGP_CUDA; b.cuda_context = context;
    tpgp_ledger x, y;
    tpgp_ledger_init(&x, validated); tpgp_ledger_init(&y, validated);
    for (int step = 0; step < 3; step++) {
        tpgp_map before_cpu = {0}, before_gpu = {0};
        if (tpgp_has_mpfr()) {
            CHECK(tpgp_map_clone(&before_cpu, &cpu) == TPGP_OK);
            CHECK(tpgp_map_clone(&before_gpu, &gpu) == TPGP_OK);
        }
        CHECK(tpgp_strang_step(&cpu, terms, NT, 1.0/64, &a, &x) == TPGP_OK);
        CHECK(tpgp_strang_step(&gpu, terms, NT, 1.0/64, &b, &y) == TPGP_OK);
        compare(&cpu, &gpu, &x, &y);
        if (tpgp_has_mpfr()) {
            double fc, rc, fg, rg, bc, bg;
            CHECK(tpgp_readout(&cpu, 1, N, &fc, &rc) == TPGP_OK);
            CHECK(tpgp_readout(&gpu, 1, N, &fg, &rg) == TPGP_OK);
            CHECK(same_double(fc, fg) && same_double(rc, rg));
            CHECK(tpgp_residual_bound(&before_cpu, &cpu, terms, NT, 1.0/64, 1, &bc) == TPGP_OK);
            CHECK(tpgp_residual_bound(&before_gpu, &gpu, terms, NT, 1.0/64, 1, &bg) == TPGP_OK);
            CHECK(same_double(bc, bg));
            tpgp_map_free(&before_cpu); tpgp_map_free(&before_gpu);
        }
    }
    printf("CUDA J1-J2 N=6: threshold=%g enclosed=%d, coefficients/ledgers/readout/residual agree\n",
           threshold, validated);
    tpgp_map_free(&cpu); tpgp_map_free(&gpu);
}
int main(int argc, char **argv) {
    int require = argc == 2 && !strcmp(argv[1], "--require-device");
    if (argc > 2 || (argc == 2 && !require)) return 2;
    char error[512];
    tpgp_cuda_context *context = tpgp_cuda_create(0, error, sizeof(error));
    if (!context) {
        fprintf(stderr, "CUDA device unavailable; no CPU substitution: %s\n", error);
        return require ? 1 : 77;
    }
    printf("CUDA full-trajectory device: %s\n", tpgp_cuda_device_name(context));
    trajectory(context, 0, 0); trajectory(context, 1e-5, 0);
    if (tpgp_has_mpfr()) {
        trajectory(context, 0, 1); trajectory(context, 1e-5, 1);
    }
    tpgp_cuda_destroy(context);
    printf("CUDA whole-trajectory checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
