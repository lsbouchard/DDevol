#ifndef TPGP_CUDA_REV_H
#define TPGP_CUDA_REV_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct tpgp_cuda_context tpgp_cuda_context;
typedef enum {
    TPGP_CUDA_OK = 0,
    TPGP_CUDA_EINVAL,
    TPGP_CUDA_ENOMEM,
    TPGP_CUDA_EDEVICE,
    TPGP_CUDA_ENUMERIC
} tpgp_cuda_status;

/* compiled() reports build capability, not runtime device availability.
 * Device indices follow CUDA_VISIBLE_DEVICES. No automatic CPU fallback. */
int tpgp_cuda_compiled(void);
tpgp_cuda_context *tpgp_cuda_create(int device_index, char *error, size_t error_capacity);
void tpgp_cuda_destroy(tpgp_cuda_context *context);
const char *tpgp_cuda_device_name(const tpgp_cuda_context *context);

/* Generate the two raw local-rotation branches for each input coefficient.
 * Keys use I=0,X=1,Y=2,Z=3 in two bits/site, site s in word s/32.
 * Output slots 2*k and 2*k+1 match the serial engine's deterministic order.
 * Commuting P gives (P,v,0) and (P,0,0); anticommuting P gives
 * (P, RN(v*c), radius_c) and (key(-i QP), sign(-i QP)*RN(v*s), radius_s).
 * Host-provided c,s,rc,rs are the same enclosed trigonometric factors used by
 * the CPU engine; no trigonometric approximation is performed on the GPU.
 * If enclose==0 all output radii are zero. Otherwise the same separately
 * rounded binary64 radius formulas as the serial engine are evaluated.
 *
 * Arrays: count*n_words input keys, count input values, 2*count*n_words
 * output keys, and 2*count output values/radii. All arrays must be disjoint.
 * n_words=ceil(n_spins/32), distinct valid i,j, qi,qj in [0,3], finite
 * coefficients/factors, nonnegative finite radii, and zero padded key bits.
 * Empty input accepts NULL arrays. Output is usable only after success.
 * Calls sharing a context are serialized; destruction must not overlap use.
 * Device allocations persist across calls and grow as needed. */
tpgp_cuda_status tpgp_cuda_branches(
    tpgp_cuda_context *context,
    const uint64_t *keys, const double *values, size_t count,
    size_t n_words, size_t n_spins, size_t i, size_t j, unsigned qi, unsigned qj,
    double c, double s, double rc, double rs, int enclose,
    uint64_t *output_keys, double *output_values, double *output_radii,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
