#include "tpgp_cuda_rev.h"
#include <stdio.h>
static void unavailable(char *error, size_t size) {
    if (error && size) snprintf(error, size, "CUDA was not enabled in this build; choose a CPU backend explicitly or rebuild with TPGP_ENABLE_CUDA=ON.");
}
int tpgp_cuda_compiled(void) { return 0; }
tpgp_cuda_context *tpgp_cuda_create(int index, char *error, size_t size) {
    (void)index; unavailable(error, size); return NULL;
}
void tpgp_cuda_destroy(tpgp_cuda_context *context) { (void)context; }
const char *tpgp_cuda_device_name(const tpgp_cuda_context *context) {
    (void)context; return "unavailable";
}
tpgp_cuda_status tpgp_cuda_branches(tpgp_cuda_context *context,
    const uint64_t *keys, const double *values, size_t count,
    size_t words, size_t spins, size_t i, size_t j, unsigned qi, unsigned qj,
    double c, double s, double rc, double rs, int enclose,
    uint64_t *output_keys, double *output_values, double *output_radii,
    char *error, size_t size) {
    (void)context; (void)keys; (void)values; (void)count; (void)words; (void)spins;
    (void)i; (void)j; (void)qi; (void)qj; (void)c; (void)s; (void)rc; (void)rs;
    (void)enclose; (void)output_keys; (void)output_values; (void)output_radii;
    unavailable(error, size); return TPGP_CUDA_EDEVICE;
}
