#ifndef TPGP_METAL_REV_H
#define TPGP_METAL_REV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exact integer-only accelerator. Coefficient arithmetic remains on the CPU.
 * Packed keys use I=0, X=1, Y=2, Z=3; site s is bits 2*(s%32)
 * in word s/32. No Hilbert-space vector or FP32 coefficients are created. */
typedef struct tpgp_metal_context tpgp_metal_context;

/* Returns NULL and a diagnostic if Metal is unavailable. Never falls back. */
tpgp_metal_context *tpgp_metal_create(char *error, size_t error_capacity);
void tpgp_metal_destroy(tpgp_metal_context *context);
const char *tpgp_metal_device_name(tpgp_metal_context *context);

/* For each P, test anticommutation with Q=qi(i) qj(j). If anticommuting,
 * branch_keys and branch_sign encode the Hermitian Pauli string -i QP.
 * If commuting, the input key is copied and branch_sign=0. All output arrays
 * must be distinct from the input and from one another. Valid zero-count
 * calls accept NULL arrays. Returns 0 on success, -1 on any failure.
 * n_words must equal ceil(n_spins/32), and i and j must be distinct sites.
 * qi and qj may include identity. Calls sharing a context are serialized.
 * Context destruction must not overlap a call using that context. */
int tpgp_metal_transform(tpgp_metal_context *context,
                         const uint64_t *keys, size_t count,
                         size_t n_words, size_t n_spins,
                         size_t i, size_t j, unsigned qi, unsigned qj,
                         uint64_t *branch_keys, int8_t *branch_sign,
                         uint8_t *anticommute,
                         char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
