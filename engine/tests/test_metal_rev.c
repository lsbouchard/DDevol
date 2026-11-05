#include "tpgp_metal_rev.h"
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t random_word(uint64_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

/* Independent scalar specification: multiplication phase from cyclic order,
 * not from the shader's lookup table. */
static unsigned exponent(unsigned q, unsigned p) {
    if (!q || !p || q == p) return 0;
    return (q == 1 && p == 2) || (q == 2 && p == 3) ||
           (q == 3 && p == 1) ? 1 : 3;
}

static unsigned symbol(const uint64_t *key, size_t site) {
    return (unsigned)((key[site / 32] >> (2 * (site % 32))) & UINT64_C(3));
}

static void reference(const uint64_t *input, size_t words,
                       size_t i, size_t j, unsigned qi, unsigned qj,
                       uint64_t *output, int8_t *sign, uint8_t *anti) {
    unsigned pi = symbol(input, i), pj = symbol(input, j);
    unsigned phase = (exponent(qi, pi) + exponent(qj, pj)) % 4;
    *anti = (uint8_t)(phase % 2);
    *sign = *anti ? (phase == 1 ? 1 : -1) : 0;
    memcpy(output, input, words * sizeof(uint64_t));
    if (*anti) {
        output[i / 32] ^= (uint64_t)qi << (2 * (i % 32));
        output[j / 32] ^= (uint64_t)qj << (2 * (j % 32));
    }
}

static int compare_batch(tpgp_metal_context *context,
                         const uint64_t *input, size_t count,
                         size_t words, size_t spins, size_t i, size_t j,
                         unsigned qi, unsigned qj) {
    uint64_t *output = calloc(count * words, sizeof(*output));
    uint64_t *expected = calloc(words, sizeof(*expected));
    int8_t *sign = calloc(count, sizeof(*sign));
    uint8_t *anti = calloc(count, sizeof(*anti));
    char error[512];
    if (!output || !expected || !sign || !anti) {
        fprintf(stderr, "Test allocation failed.\n");
        free(output); free(expected); free(sign); free(anti);
        return 1;
    }
    int failed = tpgp_metal_transform(context, input, count, words, spins,
                                     i, j, qi, qj, output, sign, anti,
                                     error, sizeof(error)) != 0;
    if (failed) fprintf(stderr, "Metal transform failed: %s\n", error);
    for (size_t k = 0; !failed && k < count; ++k) {
        int8_t wanted_sign;
        uint8_t wanted_anti;
        reference(input + k * words, words, i, j, qi, qj,
                   expected, &wanted_sign, &wanted_anti);
        if (memcmp(expected, output + k * words, words * sizeof(*output)) ||
            sign[k] != wanted_sign || anti[k] != wanted_anti) {
            fprintf(stderr, "Mismatch: N=%zu, key=%zu, sites=(%zu,%zu), Q=(%u,%u), sign=%d/%d, anti=%u/%u\n",
                    spins, k, i, j, qi, qj, sign[k], wanted_sign, anti[k], wanted_anti);
            failed = 1;
        }
    }
    free(output); free(expected); free(sign); free(anti);
    return failed;
}

typedef struct { tpgp_metal_context *context; uint64_t seed; int failed; } ThreadCase;

static void *concurrent_case(void *opaque) {
    ThreadCase *test = opaque;
    uint64_t keys[64 * 32];
    for (size_t k = 0; k < 64 * 32; ++k) keys[k] = random_word(&test->seed);
    for (unsigned q = 1; q <= 3 && !test->failed; ++q)
        test->failed = compare_batch(test->context, keys, 64, 32, 1000,
                                     511, 999, q, q);
    return NULL;
}

int main(void) {
    char error[512];
    tpgp_metal_context *context = tpgp_metal_create(error, sizeof(error));
    if (!context) {
        fprintf(stderr, "SKIP: real Metal device unavailable: %s\n", error);
        return 77;
    }
    printf("Metal device: %s\n", tpgp_metal_device_name(context));
    uint64_t small[16];
    for (unsigned k = 0; k < 16; ++k) small[k] = k;
    int failed = 0;
    for (unsigned qi = 0; qi < 4 && !failed; ++qi)
        for (unsigned qj = 0; qj < 4 && !failed; ++qj)
            failed = compare_batch(context, small, 16, 1, 2, 0, 1, qi, qj);
    if (!failed) puts("PASS: all 256 two-site Pauli generator/string combinations.");

    enum { COUNT = 257, WORDS = 32, SPINS = 1000 };
    uint64_t *large = malloc(COUNT * WORDS * sizeof(*large));
    if (!large) failed = 1;
    uint64_t seed = UINT64_C(0x9e3779b97f4a7c15);
    if (large) for (size_t k = 0; k < COUNT * WORDS; ++k) large[k] = random_word(&seed);
    const size_t pairs[][2] = {{0, 15}, {15, 16}, {31, 32}, {511, 999}, {999, 0}};
    for (size_t p = 0; p < sizeof(pairs) / sizeof(pairs[0]) && !failed; ++p)
        for (unsigned qi = 0; qi < 4 && !failed; ++qi)
            for (unsigned qj = 0; qj < 4 && !failed; ++qj)
                failed = compare_batch(context, large, COUNT, WORDS, SPINS,
                                         pairs[p][0], pairs[p][1], qi, qj);
    if (!failed) puts("PASS: 20,560 N=1000 transforms; 32/64-bit boundaries and last site; bitwise exact keys/signs.");
    free(large);

    uint64_t input = 1, output = 0;
    int8_t sign = 0;
    uint8_t anti = 0;
#define EXPECT_REJECT(label, ctx, in, count, words, spins, i, j, qi, qj, out, sg, an) \
    do { \
        if (tpgp_metal_transform(ctx, in, count, words, spins, i, j, qi, qj, \
                                 out, sg, an, error, sizeof(error)) == 0) { \
            fprintf(stderr, "Invalid input accepted: %s\n", label); failed = 1; \
        } \
    } while (0)
    EXPECT_REJECT("null context", NULL, &input, 1, 1, 2, 0, 1, 1, 1, &output, &sign, &anti);
    EXPECT_REJECT("same site", context, &input, 1, 1, 2, 0, 0, 1, 1, &output, &sign, &anti);
    EXPECT_REJECT("invalid Pauli", context, &input, 1, 1, 2, 0, 1, 4, 1, &output, &sign, &anti);
    EXPECT_REJECT("invalid key width", context, &input, 1, 2, 2, 0, 1, 1, 1, &output, &sign, &anti);
    EXPECT_REJECT("site out of range", context, &input, 1, 1, 2, 0, 2, 1, 1, &output, &sign, &anti);
    EXPECT_REJECT("count overflow", context, &input, SIZE_MAX, 1, 2, 0, 1, 1, 1, &output, &sign, &anti);
    EXPECT_REJECT("word-address overflow", context, &input, UINT32_MAX, 1, 2, 0, 1, 1, 1, &output, &sign, &anti);
    EXPECT_REJECT("null input", context, NULL, 1, 1, 2, 0, 1, 1, 1, &output, &sign, &anti);
    EXPECT_REJECT("overlapping arrays", context, &input, 1, 1, 2, 0, 1, 1, 1, &input, &sign, &anti);
    if (tpgp_metal_transform(context, NULL, 0, 1, 2, 0, 1, 1, 1,
                             NULL, NULL, NULL, error, sizeof(error)) != 0) failed = 1;
    if (!failed) puts("PASS: empty input and nine rejected invalid/overflow/alias inputs.");
#undef EXPECT_REJECT

    pthread_t threads[2];
    ThreadCase cases[2] = {{context, UINT64_C(0xa21f1234), 0},
                           {context, UINT64_C(0xbc342957), 0}};
    unsigned created = 0;
    for (unsigned k = 0; k < 2; ++k) {
        if (pthread_create(&threads[k], NULL, concurrent_case, &cases[k])) {
            failed = 1;
            break;
        }
        created++;
    }
    for (unsigned k = 0; k < created; ++k) {
        if (pthread_join(threads[k], NULL)) failed = 1;
        failed |= cases[k].failed;
    }
    if (!failed) puts("PASS: two simultaneous callers share one serialized Metal context.");
    tpgp_metal_destroy(context);
    if (!failed) puts("All Metal smoke checks passed; no GPU floating-point arithmetic is used.");
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
