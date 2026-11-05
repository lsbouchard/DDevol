#include "tpgp_metal_rev.h"
#include <stdio.h>

static void unavailable(char *error, size_t capacity) {
    if (error && capacity)
        snprintf(error, capacity, "Metal is not available in this platform build; select the pthread or OpenMP backend explicitly.");
}

tpgp_metal_context *tpgp_metal_create(char *error, size_t capacity) {
    unavailable(error, capacity);
    return NULL;
}

void tpgp_metal_destroy(tpgp_metal_context *context) { (void)context; }

const char *tpgp_metal_device_name(tpgp_metal_context *context) {
    (void)context;
    return "unavailable";
}

int tpgp_metal_transform(tpgp_metal_context *context,
                         const uint64_t *keys, size_t count,
                         size_t n_words, size_t n_spins,
                         size_t i, size_t j, unsigned qi, unsigned qj,
                         uint64_t *branch_keys, int8_t *branch_sign,
                         uint8_t *anticommute,
                         char *error, size_t capacity) {
    (void)context; (void)keys; (void)count; (void)n_words; (void)n_spins;
    (void)i; (void)j; (void)qi; (void)qj;
    (void)branch_keys; (void)branch_sign; (void)anticommute;
    unavailable(error, capacity);
    return -1;
}
