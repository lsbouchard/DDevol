#include "tpgp_cuda_rev.h"
#include <float.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
static uint64_t random_word(uint64_t *state) {
    *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17; return *state;
}
static double upadd(double a, double b) {
    if (a == 0) return b; if (b == 0) return a; return nextafter(a + b, INFINITY);
}
static double upmul(double a, double b) {
    if (a == 0 || b == 0) return 0; return nextafter(a * b, INFINITY);
}
static double round_radius(double x) {
    if (!isfinite(x)) return INFINITY;
    return fmax(fabs(nextafter(x, INFINITY) - x), fabs(x - nextafter(x, -INFINITY)));
}
static void reference(const uint64_t *keys, const double *values, size_t count, size_t words,
                       size_t i, size_t j, unsigned qi, unsigned qj,
                       double c, double s, double rc, double rs, int enclose,
                       uint64_t *out_keys, double *out_values, double *out_radii) {
    static const unsigned phase[4][4] = {{0,0,0,0},{0,0,1,3},{0,3,0,1},{0,1,3,0}};
    for (size_t k = 0; k < count; ++k) {
        const uint64_t *key = keys + k * words;
        uint64_t *a = out_keys + 2 * k * words, *b = a + words;
        memcpy(a, key, words * sizeof(*a)); memcpy(b, key, words * sizeof(*b));
        unsigned pi = (unsigned)(key[i/32] >> (2*(i%32))) & 3u;
        unsigned pj = (unsigned)(key[j/32] >> (2*(j%32))) & 3u;
        unsigned p = (phase[qi][pi] + phase[qj][pj]) & 3u;
        out_radii[2*k] = out_radii[2*k+1] = 0;
        if (!(p & 1)) { out_values[2*k] = values[k]; out_values[2*k+1] = 0; continue; }
        b[i/32] ^= (uint64_t)qi << (2*(i%32));
        b[j/32] ^= (uint64_t)qj << (2*(j%32));
        double x = values[k] * c, y = values[k] * s;
        out_values[2*k] = x; out_values[2*k+1] = (p == 1 ? 1 : -1) * y;
        if (enclose) {
            out_radii[2*k] = upadd(upmul(fabs(values[k]),rc),round_radius(x));
            out_radii[2*k+1] = upadd(upmul(fabs(values[k]),rs),round_radius(y));
        }
    }
}
static int compare(tpgp_cuda_context *context, const uint64_t *keys, const double *values,
                    size_t count, size_t words, size_t spins, size_t i, size_t j,
                    unsigned qi, unsigned qj, double c, double s, double rc, double rs, int enclose) {
    size_t nk = 2 * count * words, nv = 2 * count;
    uint64_t *actual_keys = malloc(nk * sizeof(uint64_t)), *wanted_keys = malloc(nk * sizeof(uint64_t));
    double *actual_values = malloc(nv * sizeof(double)), *wanted_values = malloc(nv * sizeof(double));
    double *actual_radii = malloc(nv * sizeof(double)), *wanted_radii = malloc(nv * sizeof(double));
    int failed = !actual_keys || !wanted_keys || !actual_values || !wanted_values || !actual_radii || !wanted_radii;
    if (!failed) {
        char error[512];
        reference(keys, values, count, words, i, j, qi, qj, c, s, rc, rs, enclose,
                    wanted_keys, wanted_values, wanted_radii);
        tpgp_cuda_status status = tpgp_cuda_branches(context, keys, values, count, words, spins,
            i, j, qi, qj, c, s, rc, rs, enclose, actual_keys, actual_values, actual_radii, error, sizeof(error));
        if (status != TPGP_CUDA_OK) { fprintf(stderr, "CUDA failed: %s\n", error); failed = 1; }
        if (!failed && (memcmp(actual_keys, wanted_keys, nk * sizeof(uint64_t)) ||
                        memcmp(actual_values, wanted_values, nv * sizeof(double)) ||
                        memcmp(actual_radii, wanted_radii, nv * sizeof(double)))) {
            fprintf(stderr, "Bitwise mismatch: N=%zu count=%zu sites=(%zu,%zu) Q=(%u,%u) enclose=%d\n",
                    spins, count, i, j, qi, qj, enclose);
            for (size_t k = 0; k < nv; ++k) {
                if (memcmp(actual_values+k,wanted_values+k,sizeof(double)) || memcmp(actual_radii+k,wanted_radii+k,sizeof(double))) {
                    fprintf(stderr, "slot=%zu coefficient=%a expected=%a radius=%a expected=%a\n", k,
                            actual_values[k],wanted_values[k],actual_radii[k],wanted_radii[k]); break;
                }
            }
            failed = 1;
        }
    }
    free(actual_keys); free(wanted_keys); free(actual_values); free(wanted_values); free(actual_radii); free(wanted_radii);
    return failed;
}

typedef struct { tpgp_cuda_context *context; uint64_t seed; int failed; } thread_case;
static void *concurrent(void *opaque) {
    thread_case *t = opaque;
    uint64_t keys[64*32]; double values[64];
    for (size_t k = 0; k < 64; ++k) {
        for (size_t w = 0; w < 32; ++w) keys[k*32+w] = random_word(&t->seed);
        keys[k*32+31] &= UINT64_C(0xffff);
        values[k] = (double)(k+1)/64;
    }
    for (unsigned q = 1; q <= 3 && !t->failed; ++q)
        t->failed = compare(t->context, keys, values, 64, 32, 1000, 511, 999, q, q, .6, -.8, 0x1p-54, 0x1p-54, 1);
    return NULL;
}

int main(int argc, char **argv) {
    int require_device = argc == 2 && !strcmp(argv[1], "--require-device");
    if (argc > 2 || (argc == 2 && !require_device)) { fprintf(stderr, "Usage: %s [--require-device]\n",argv[0]); return 2; }
    char error[512];
    tpgp_cuda_context *context = tpgp_cuda_create(0, error, sizeof(error));
    if (!context) {
        fprintf(stderr, "%s: CUDA compiled=%d; %s\n", require_device ? "FAIL" : "SKIP", tpgp_cuda_compiled(),error);
        return require_device ? EXIT_FAILURE : 77;
    }
    CHECK(tpgp_cuda_compiled() == 1);
    CHECK(fegetround() == FE_TONEAREST);
    printf("CUDA device: %s\n", tpgp_cuda_device_name(context));
    uint64_t keys[256]; double values[256];
    const double edge[] = {0, -0.0, DBL_TRUE_MIN, -DBL_TRUE_MIN, DBL_MIN, -DBL_MIN,
        1, -1, DBL_MAX*.25, -DBL_MAX*.25, 0x1.fffffffffffffp-1, -0x1.fffffffffffffp-1,
        0x1p-1000, -0x1p-1000, 0x1p1000, -0x1p1000};
    /* Cross every edge value with every Pauli key. Tying one value to each
     * key would leave some signed-zero/subnormal phase combinations untested. */
    for (unsigned p = 0; p < 16; ++p) for (unsigned v = 0; v < 16; ++v) {
        keys[16*p+v] = p; values[16*p+v] = edge[v];
    }
    for (unsigned qi = 0; qi < 4; ++qi) for (unsigned qj = 0; qj < 4; ++qj)
        for (int enclose = 0; enclose <= 1; ++enclose)
            CHECK(!compare(context, keys, values, 256, 1, 2, 0, 1, qi, qj, .6, -.8, 0x1p-54, 0x1p-1050, enclose));
    puts("PASS: 8192 two-site branch cases; every Pauli key and edge coefficient crossed, including signed zero and subnormals.");

    enum {COUNT=257, WORDS=32};
    uint64_t *large = malloc(COUNT*WORDS*sizeof(uint64_t)); double *coeff = malloc(COUNT*sizeof(double));
    CHECK(large && coeff);
    uint64_t seed = UINT64_C(0x9e3779b97f4a7c15);
    for (size_t k = 0; k < COUNT; ++k) {
        for (size_t w = 0; w < WORDS; ++w) large[k*WORDS+w] = random_word(&seed);
        large[k*WORDS+WORDS-1] &= UINT64_C(0xffff);
        coeff[k] = k < 16 ? edge[k] : ldexp((double)(random_word(&seed)>>11)*0x1p-53,(int)(k%1800)-900);
    }
    const size_t pairs[][2] = {{0,15},{15,16},{31,32},{511,999},{999,0}};
    for (size_t p = 0; p < sizeof(pairs)/sizeof(pairs[0]); ++p)
        for (unsigned qi = 0; qi < 4; ++qi) for (unsigned qj = 0; qj < 4; ++qj)
            CHECK(!compare(context,large,coeff,COUNT,WORDS,1000,pairs[p][0],pairs[p][1],qi,qj,-.8,.6,0x1p-54,0x1p-54,1));
    CHECK(!compare(context,large,coeff,COUNT,WORDS,1000,0,999,3,3,1,-0.0,0,0,1));
    CHECK(!compare(context,large,coeff,COUNT,WORDS,1000,0,999,3,3,0,-1,0,0,1));
    free(large); free(coeff);
    puts("PASS: 20,560 N=1000 branch cases plus exact-angle factors; all output arrays bitwise equal.");

    uint64_t input_key = 1, output_keys[2] = {0,0};
    double input_value = 1, output_values[2] = {0,0}, output_radii[2] = {0,0};
#define CALL(ctx,k,v,n,w,sp,i,j,qi,qj,c,s,rc,rs,en,outk,outv,outr) \
    tpgp_cuda_branches(ctx,k,v,n,w,sp,i,j,qi,qj,c,s,rc,rs,en,outk,outv,outr,error,sizeof(error))
#define VALID_CALL CALL(context,&input_key,&input_value,1,1,2,0,1,3,3,.6,.8,0,0,1,output_keys,output_values,output_radii)
    CHECK(CALL(NULL,&input_key,&input_value,1,1,2,0,1,3,3,.6,.8,0,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,NULL,&input_value,1,1,2,0,1,3,3,.6,.8,0,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,SIZE_MAX,1,2,0,1,3,3,.6,.8,0,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,1,2,2,0,1,3,3,.6,.8,0,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,1,1,2,0,0,3,3,.6,.8,0,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,1,1,2,0,1,4,3,.6,.8,0,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,1,1,2,0,1,3,3,NAN,.8,0,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,1,1,2,0,1,3,3,.6,.8,INFINITY,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,1,1,2,0,1,3,3,.6,.8,-1,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,1,1,2,0,1,3,3,.6,.8,0,0,2,output_keys,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,&input_key,&input_value,1,1,2,0,1,3,3,.6,.8,0,0,1,&input_key,output_values,output_radii)==TPGP_CUDA_EINVAL);
    CHECK(CALL(context,NULL,NULL,0,1,2,0,1,3,3,.6,.8,0,0,1,NULL,NULL,NULL)==TPGP_CUDA_OK);
    input_key = 16; CHECK(VALID_CALL == TPGP_CUDA_EINVAL); input_key = 1;
    input_value = NAN; CHECK(VALID_CALL == TPGP_CUDA_ENUMERIC);
    input_value = INFINITY; CHECK(VALID_CALL == TPGP_CUDA_ENUMERIC);
    /* Nonfinite commuting inputs must also fail, not take a bypass around
     * the arithmetic checks. */
    input_key = 0; CHECK(VALID_CALL == TPGP_CUDA_ENUMERIC); input_key = 1;
    input_value = DBL_MAX;
    CHECK(CALL(context,&input_key,&input_value,1,1,2,0,1,3,3,2,.8,0,0,1,output_keys,output_values,output_radii)==TPGP_CUDA_ENUMERIC);
    input_value = 1; CHECK(VALID_CALL == TPGP_CUDA_OK);
    CHECK(tpgp_cuda_create(-1,error,sizeof(error)) == NULL);
    CHECK(tpgp_cuda_create(INT_MAX,error,sizeof(error)) == NULL);
#undef VALID_CALL
#undef CALL
    puts("PASS: invalid inputs, padding, overflow, nonfinite arithmetic, empty batch, and reuse after failure.");

    pthread_t threads[2]; thread_case cases[2] = {{context,UINT64_C(0x431217),0},{context,UINT64_C(0x187721),0}};
    unsigned made = 0;
    for (unsigned k = 0; k < 2; ++k) { if (pthread_create(threads+k,NULL,concurrent,cases+k)) break; made++; }
    for (unsigned k = 0; k < made; ++k) CHECK(!pthread_join(threads[k],NULL));
    CHECK(made == 2 && !cases[0].failed && !cases[1].failed);
    tpgp_cuda_destroy(context);
    printf("PASS: two concurrent callers; %u host checks; actual GPU binary64 branch suite complete.\n",checks);
    return 0;
}
