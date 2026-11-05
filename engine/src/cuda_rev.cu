#include "tpgp_cuda_rev.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && FLT_RADIX == 2,
              "The CUDA branch backend requires IEEE binary64.");

struct tpgp_cuda_context {
    int device;
    int max_blocks;
    cudaStream_t stream;
    void *storage;
    size_t capacity;
    pthread_mutex_t mutex;
    char device_name[256];
};

static void message(char *error, size_t size, const char *text) {
    if (error && size) snprintf(error, size, "%s", text);
}
static tpgp_cuda_status cuda_failure(cudaError_t code, const char *operation,
                                     char *error, size_t size) {
    if (error && size) snprintf(error, size, "%s: %s", operation, cudaGetErrorString(code));
    return code == cudaErrorMemoryAllocation ? TPGP_CUDA_ENOMEM : TPGP_CUDA_EDEVICE;
}
static int product(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b) return 0;
    *out = a * b; return 1;
}
static int sum(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return 0;
    *out = a + b; return 1;
}
static int overlap(const void *a, size_t as, const void *b, size_t bs) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (as > UINTPTR_MAX - x || bs > UINTPTR_MAX - y) return 1;
    return x < y + bs && y < x + as;
}

/* Exact adjacent binary64 value, including zero/subnormal transitions. No
 * approximate transcendental function or single-precision operand is used. */
__device__ static double next_up(double x) {
    if (isnan(x) || x == CUDART_INF) return x;
    if (x == 0) return __longlong_as_double(1LL);
    unsigned long long bits = (unsigned long long)__double_as_longlong(x);
    bits = x > 0 ? bits + 1 : bits - 1;
    return __longlong_as_double((long long)bits);
}
__device__ static double next_down(double x) { return -next_up(-x); }
__device__ static double upadd(double a, double b) {
    if (a == 0) return b;
    if (b == 0) return a;
    return next_up(__dadd_rn(a, b));
}
__device__ static double upmul(double a, double b) {
    if (a == 0 || b == 0) return 0;
    return next_up(__dmul_rn(a, b));
}
__device__ static double round_radius(double x) {
    if (!isfinite(x)) return CUDART_INF;
    double a = fabs(__dsub_rn(next_up(x), x));
    double b = fabs(__dsub_rn(x, next_down(x)));
    return a > b ? a : b;
}
__device__ static unsigned pauli_phase(unsigned q, unsigned p) {
    if (!q || !p || q == p) return 0;
    return (q == 1 && p == 2) || (q == 2 && p == 3) || (q == 3 && p == 1) ? 1 : 3;
}

__global__ static void branch_kernel(
    const uint64_t *keys, const double *values, size_t count, size_t words,
    size_t spins, size_t i, size_t j, unsigned qi, unsigned qj,
    double c, double s, double rc, double rs, int enclose,
    uint64_t *out_keys, double *out_values, double *out_radii, int *failure) {
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < count; k += stride) {
        const uint64_t *key = keys + k * words;
        uint64_t *a = out_keys + (2 * k) * words, *d = a + words;
        size_t rem = spins % 32;
        if (rem && key[words - 1] >> (2 * rem)) atomicOr(failure, 1);
        for (size_t w = 0; w < words; ++w) a[w] = d[w] = key[w];
        unsigned pi = (unsigned)((key[i / 32] >> (2 * (i % 32))) & 3u);
        unsigned pj = (unsigned)((key[j / 32] >> (2 * (j % 32))) & 3u);
        bool anti = (qi && pi && qi != pi) != (qj && pj && qj != pj);
        double v = values[k];
        if (!isfinite(v)) atomicOr(failure, 2);
        out_radii[2 * k] = out_radii[2 * k + 1] = 0;
        if (!anti) {
            out_values[2 * k] = v;
            out_values[2 * k + 1] = 0;
            continue;
        }
        unsigned phase = (pauli_phase(qi, pi) + pauli_phase(qj, pj)) & 3u;
        double sign = phase == 1 ? 1.0 : -1.0;
        d[i / 32] ^= (uint64_t)qi << (2 * (i % 32));
        d[j / 32] ^= (uint64_t)qj << (2 * (j % 32));
        double x = __dmul_rn(v, c), y = __dmul_rn(v, s);
        out_values[2 * k] = x;
        out_values[2 * k + 1] = __dmul_rn(sign, y);
        if (enclose) {
            out_radii[2 * k] = upadd(upmul(fabs(v), rc), round_radius(x));
            out_radii[2 * k + 1] = upadd(upmul(fabs(v), rs), round_radius(y));
        }
        if (!isfinite(x) || !isfinite(y) || !isfinite(out_radii[2 * k]) ||
            !isfinite(out_radii[2 * k + 1])) atomicOr(failure, 2);
    }
}

extern "C" int tpgp_cuda_compiled(void) { return 1; }
extern "C" tpgp_cuda_context *tpgp_cuda_create(int index, char *error, size_t size) {
    message(error, size, "");
    int count = 0, previous = 0;
    cudaError_t code = cudaGetDeviceCount(&count);
    if (code != cudaSuccess) { cuda_failure(code, "CUDA device discovery", error, size); return NULL; }
    if (index < 0 || index >= count) { message(error, size, "Requested CUDA device index is unavailable."); return NULL; }
    code = cudaGetDevice(&previous);
    if (code != cudaSuccess) { cuda_failure(code, "CUDA current-device query", error, size); return NULL; }
    code = cudaSetDevice(index);
    if (code != cudaSuccess) { cuda_failure(code, "CUDA device selection", error, size); return NULL; }
    cudaDeviceProp property;
    code = cudaGetDeviceProperties(&property, index);
    if (code != cudaSuccess || property.major < 2) {
        if (code != cudaSuccess) cuda_failure(code, "CUDA device properties", error, size);
        else message(error, size, "CUDA compute capability 2.0 or later is required for this binary64 backend.");
        cudaSetDevice(previous); return NULL;
    }
    tpgp_cuda_context *context = (tpgp_cuda_context *)calloc(1, sizeof(*context));
    if (!context) { message(error, size, "Could not allocate the CUDA host context."); cudaSetDevice(previous); return NULL; }
    if (pthread_mutex_init(&context->mutex, NULL)) {
        message(error, size, "Could not initialize the CUDA context mutex.");
        free(context); cudaSetDevice(previous); return NULL;
    }
    context->device = index;
    context->max_blocks = property.multiProcessorCount > INT_MAX/32 ? INT_MAX :
                          property.multiProcessorCount > 0 ? property.multiProcessorCount * 32 : 32;
    if (context->max_blocks > property.maxGridSize[0]) context->max_blocks = property.maxGridSize[0];
    snprintf(context->device_name, sizeof(context->device_name), "%s (sm_%d%d)", property.name, property.major, property.minor);
    code = cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking);
    if (code != cudaSuccess) {
        cuda_failure(code, "CUDA stream creation", error, size);
        pthread_mutex_destroy(&context->mutex); free(context); cudaSetDevice(previous); return NULL;
    }
    code = cudaSetDevice(previous);
    if (code != cudaSuccess) {
        cuda_failure(code, "CUDA previous-device restoration", error, size);
        cudaStreamDestroy(context->stream); pthread_mutex_destroy(&context->mutex); free(context); return NULL;
    }
    return context;
}
extern "C" void tpgp_cuda_destroy(tpgp_cuda_context *context) {
    if (!context) return;
    int previous = context->device;
    cudaGetDevice(&previous);
    cudaSetDevice(context->device);
    cudaStreamSynchronize(context->stream);
    if (context->storage) cudaFree(context->storage);
    cudaStreamDestroy(context->stream);
    cudaSetDevice(previous);
    pthread_mutex_destroy(&context->mutex);
    free(context);
}
extern "C" const char *tpgp_cuda_device_name(const tpgp_cuda_context *context) {
    return context ? context->device_name : "unavailable";
}

extern "C" tpgp_cuda_status tpgp_cuda_branches(
    tpgp_cuda_context *context, const uint64_t *keys, const double *values, size_t count,
    size_t words, size_t spins, size_t i, size_t j, unsigned qi, unsigned qj,
    double c, double s, double rc, double rs, int enclose,
    uint64_t *out_keys, double *out_values, double *out_radii, char *error, size_t size) {
    message(error, size, "");
    if (!context || spins < 2 || words != spins / 32 + (spins % 32 != 0) ||
        i >= spins || j >= spins || i == j || qi > 3 || qj > 3 ||
        !isfinite(c) || !isfinite(s) || !isfinite(rc) || !isfinite(rs) || rc < 0 || rs < 0 ||
        (enclose != 0 && enclose != 1)) {
        message(error, size, "Invalid CUDA branch dimensions, sites, generator, or trigonometric enclosure.");
        return TPGP_CUDA_EINVAL;
    }
    if (!count) return TPGP_CUDA_OK;
    size_t key_bytes, value_bytes, out_key_bytes, out_value_bytes, total, temporary;
    if (!keys || !values || !out_keys || !out_values || !out_radii ||
        !product(count, words, &temporary) || !product(temporary, sizeof(uint64_t), &key_bytes) ||
        !product(count, sizeof(double), &value_bytes) || !product(key_bytes, 2, &out_key_bytes) ||
        !product(value_bytes, 2, &out_value_bytes) || !product(key_bytes, 3, &total) ||
        !product(value_bytes, 5, &temporary) || !sum(total, temporary, &total) ||
        !sum(total, sizeof(int), &total)) {
        message(error, size, "Null CUDA array or packed-branch allocation size overflow.");
        return TPGP_CUDA_EINVAL;
    }
    const void *arrays[] = {keys, values, out_keys, out_values, out_radii};
    const size_t lengths[] = {key_bytes, value_bytes, out_key_bytes, out_value_bytes, out_value_bytes};
    for (unsigned a = 0; a < 5; ++a) for (unsigned b = a + 1; b < 5; ++b)
        if (overlap(arrays[a], lengths[a], arrays[b], lengths[b])) {
            message(error, size, "CUDA input/output arrays must be disjoint."); return TPGP_CUDA_EINVAL;
        }
    if (pthread_mutex_lock(&context->mutex)) {
        message(error, size, "Could not lock the CUDA context."); return TPGP_CUDA_EDEVICE;
    }
    tpgp_cuda_status result = TPGP_CUDA_OK;
    int previous = context->device;
    cudaError_t code = cudaGetDevice(&previous);
    if (code != cudaSuccess) result = cuda_failure(code, "CUDA current-device query", error, size);
    if (result == TPGP_CUDA_OK) {
        code = cudaSetDevice(context->device);
        if (code != cudaSuccess) result = cuda_failure(code, "CUDA device selection", error, size);
    }
    if (result == TPGP_CUDA_OK && total > context->capacity) {
        /* No state lives on the GPU between calls. Releasing old scratch first
         * avoids a transient double allocation when a trajectory grows. */
        if (context->storage) {
            code = cudaFree(context->storage);
            if (code != cudaSuccess) result = cuda_failure(code, "CUDA scratch release", error, size);
            context->storage = NULL; context->capacity = 0;
        }
        if (result == TPGP_CUDA_OK) {
            code = cudaMalloc(&context->storage, total);
            if (code != cudaSuccess) result = cuda_failure(code, "CUDA persistent scratch allocation", error, size);
            else context->capacity = total;
        }
    }
    if (result == TPGP_CUDA_OK) {
        unsigned char *base = (unsigned char *)context->storage;
        uint64_t *device_keys = (uint64_t *)base;
        double *device_values = (double *)(base + key_bytes);
        uint64_t *device_out_keys = (uint64_t *)(base + key_bytes + value_bytes);
        double *device_out_values = (double *)(base + key_bytes + value_bytes + out_key_bytes);
        double *device_out_radii = device_out_values + 2 * count;
        int *device_failure = (int *)(device_out_radii + 2 * count);
        int failure = 0;
#define CUDA_CALL(call, operation) do { if (result == TPGP_CUDA_OK) { \
    code = (call); if (code != cudaSuccess) result = cuda_failure(code, operation, error, size); \
} } while (0)
        CUDA_CALL(cudaMemcpyAsync(device_keys, keys, key_bytes, cudaMemcpyHostToDevice, context->stream), "CUDA input-key transfer");
        CUDA_CALL(cudaMemcpyAsync(device_values, values, value_bytes, cudaMemcpyHostToDevice, context->stream), "CUDA input-coefficient transfer");
        CUDA_CALL(cudaMemsetAsync(device_failure, 0, sizeof(int), context->stream), "CUDA status reset");
        if (result == TPGP_CUDA_OK) {
            size_t wanted = count / 256 + (count % 256 != 0);
            unsigned blocks = wanted < (size_t)context->max_blocks ? (unsigned)wanted : (unsigned)context->max_blocks;
            branch_kernel<<<blocks, 256, 0, context->stream>>>(device_keys, device_values, count, words,
                spins, i, j, qi, qj, c, s, rc, rs, enclose, device_out_keys,
                device_out_values, device_out_radii, device_failure);
            CUDA_CALL(cudaGetLastError(), "CUDA branch-kernel launch");
        }
        CUDA_CALL(cudaMemcpyAsync(&failure, device_failure, sizeof(int), cudaMemcpyDeviceToHost, context->stream), "CUDA status transfer");
        CUDA_CALL(cudaStreamSynchronize(context->stream), "CUDA branch-kernel execution");
        if (result == TPGP_CUDA_OK && failure) {
            result = failure & 1 ? TPGP_CUDA_EINVAL : TPGP_CUDA_ENUMERIC;
            message(error, size, failure & 1 ? "CUDA input keys contain nonzero padding bits." : "CUDA branch arithmetic or coefficients are nonfinite.");
        }
        CUDA_CALL(cudaMemcpyAsync(out_keys, device_out_keys, out_key_bytes, cudaMemcpyDeviceToHost, context->stream), "CUDA output-key transfer");
        CUDA_CALL(cudaMemcpyAsync(out_values, device_out_values, out_value_bytes, cudaMemcpyDeviceToHost, context->stream), "CUDA output-coefficient transfer");
        CUDA_CALL(cudaMemcpyAsync(out_radii, device_out_radii, out_value_bytes, cudaMemcpyDeviceToHost, context->stream), "CUDA output-radius transfer");
        CUDA_CALL(cudaStreamSynchronize(context->stream), "CUDA output completion");
        /* Drain any successfully queued operations before borrowed host arrays
         * may be released, including paths with an earlier API failure. */
        if (result != TPGP_CUDA_OK) cudaStreamSynchronize(context->stream);
#undef CUDA_CALL
    }
    code = cudaSetDevice(previous);
    if (result == TPGP_CUDA_OK && code != cudaSuccess)
        result = cuda_failure(code, "CUDA previous-device restoration", error, size);
    pthread_mutex_unlock(&context->mutex);
    return result;
}
