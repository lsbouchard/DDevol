#include "tpgp_metal_rev.h"
#include "shader_source_rev.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint32_t count, words32, site_i, site_j, qi, qj;
} PauliParameters;

@interface TPGPMetalEngine : NSObject {
@public
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> inputBuffer, outputBuffer, signBuffer, antiBuffer;
    size_t keyCapacity, countCapacity;
    pthread_mutex_t mutex;
    BOOL mutexReady;
    char deviceName[256];
}
@end

@implementation TPGPMetalEngine
- (void)dealloc {
    if (mutexReady) pthread_mutex_destroy(&mutex);
}
@end

static void diagnostic(char *error, size_t capacity, const char *message) {
    if (error && capacity) snprintf(error, capacity, "%s", message);
}

static void metal_diagnostic(char *error, size_t capacity,
                             const char *operation, NSError *detail) {
    if (error && capacity)
        snprintf(error, capacity, "%s: %s", operation,
                 detail ? detail.localizedDescription.UTF8String : "unknown Metal error");
}

tpgp_metal_context *tpgp_metal_create(char *error, size_t capacity) {
    @autoreleasepool {
        diagnostic(error, capacity, "");
        TPGPMetalEngine *engine = [[TPGPMetalEngine alloc] init];
        if (!engine) {
            diagnostic(error, capacity, "Could not allocate the Metal context.");
            return NULL;
        }
        engine->device = MTLCreateSystemDefaultDevice();
        if (!engine->device) {
            diagnostic(error, capacity, "No Metal device is available; the requested backend was not replaced by a CPU backend.");
            return NULL;
        }
        /* The shader views packed uint64_t keys as little-endian uint32_t.
         * This is the native ordering of supported Apple Silicon hosts. */
        const uint32_t endian = 1;
        if (*(const uint8_t *)&endian != 1) {
            diagnostic(error, capacity, "The Metal packed-key backend requires a little-endian host.");
            return NULL;
        }
        if (pthread_mutex_init(&engine->mutex, NULL) != 0) {
            diagnostic(error, capacity, "Could not initialize the Metal-context mutex.");
            return NULL;
        }
        engine->mutexReady = YES;
        engine->queue = [engine->device newCommandQueue];
        if (!engine->queue) {
            diagnostic(error, capacity, "Could not allocate the Metal command queue.");
            return NULL;
        }
        NSError *detail = nil;
        NSString *source = [NSString stringWithUTF8String:tpgp_metal_shader_source_rev];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        /* The shader is integer-only; floating-point compiler options cannot
         * affect its result. In particular, it contains no FP32 coefficients. */
        id<MTLLibrary> library = [engine->device newLibraryWithSource:source
                                                           options:options error:&detail];
        if (!library) {
            metal_diagnostic(error, capacity, "Metal shader compilation failed", detail);
            return NULL;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"tpgp_pauli_transform"];
        if (!function) {
            diagnostic(error, capacity, "The Metal shader entry point is missing.");
            return NULL;
        }
        engine->pipeline = [engine->device newComputePipelineStateWithFunction:function error:&detail];
        if (!engine->pipeline) {
            metal_diagnostic(error, capacity, "Metal pipeline creation failed", detail);
            return NULL;
        }
        snprintf(engine->deviceName, sizeof(engine->deviceName), "%s", engine->device.name.UTF8String);
        return (__bridge_retained tpgp_metal_context *)engine;
    }
}

void tpgp_metal_destroy(tpgp_metal_context *context) {
    if (!context) return;
    @autoreleasepool {
        TPGPMetalEngine *engine = (__bridge_transfer TPGPMetalEngine *)context;
        (void)engine;
    }
}

const char *tpgp_metal_device_name(tpgp_metal_context *context) {
    if (!context) return "unavailable";
    TPGPMetalEngine *engine = (__bridge TPGPMetalEngine *)context;
    return engine->deviceName;
}

static int overlaps(const void *a, size_t as, const void *b, size_t bs) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (as > UINTPTR_MAX - x || bs > UINTPTR_MAX - y) return 1;
    return x < y + bs && y < x + as;
}

int tpgp_metal_transform(tpgp_metal_context *context,
                         const uint64_t *keys, size_t count,
                         size_t n_words, size_t n_spins,
                         size_t i, size_t j, unsigned qi, unsigned qj,
                         uint64_t *branch_keys, int8_t *branch_sign,
                         uint8_t *anticommute,
                         char *error, size_t capacity) {
    diagnostic(error, capacity, "");
    if (!context || n_spins < 2 || n_spins > UINT32_MAX ||
        n_words != (n_spins / 32 + (n_spins % 32 != 0)) ||
        i >= n_spins || j >= n_spins || i == j || qi > 3 || qj > 3 ||
        count > UINT32_MAX || n_words > UINT32_MAX / 2) {
        diagnostic(error, capacity, "Invalid Metal Pauli-transform dimensions or generator.");
        return -1;
    }
    if (!count) return 0;
    if (!keys || !branch_keys || !branch_sign || !anticommute ||
        count > SIZE_MAX / sizeof(uint64_t) / n_words ||
        count > UINT32_MAX / (2 * n_words)) {
        diagnostic(error, capacity, "Null Metal array or packed-key size exceeds supported address range.");
        return -1;
    }
    size_t keyBytes = count * n_words * sizeof(uint64_t);
    if (overlaps(keys, keyBytes, branch_keys, keyBytes) ||
        overlaps(keys, keyBytes, branch_sign, count) ||
        overlaps(keys, keyBytes, anticommute, count) ||
        overlaps(branch_keys, keyBytes, branch_sign, count) ||
        overlaps(branch_keys, keyBytes, anticommute, count) ||
        overlaps(branch_sign, count, anticommute, count)) {
        diagnostic(error, capacity, "Metal input and output arrays must not overlap.");
        return -1;
    }
    @autoreleasepool {
        TPGPMetalEngine *engine = (__bridge TPGPMetalEngine *)context;
        if (keyBytes > engine->device.maxBufferLength || count > engine->device.maxBufferLength) {
            diagnostic(error, capacity, "Metal transform exceeds the device's maximum buffer length.");
            return -1;
        }
        if (pthread_mutex_lock(&engine->mutex) != 0) {
            diagnostic(error, capacity, "Could not lock the Metal context.");
            return -1;
        }
        int result = -1;
        if (keyBytes > engine->keyCapacity) {
            /* Only replace cached buffers after both allocations succeed. */
            id<MTLBuffer> input = [engine->device newBufferWithLength:keyBytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output = [engine->device newBufferWithLength:keyBytes options:MTLResourceStorageModeShared];
            if (!input || !output) {
                diagnostic(error, capacity, "Could not allocate Metal packed-key buffers.");
                goto done;
            }
            engine->inputBuffer = input;
            engine->outputBuffer = output;
            engine->keyCapacity = keyBytes;
        }
        if (count > engine->countCapacity) {
            id<MTLBuffer> signs = [engine->device newBufferWithLength:count options:MTLResourceStorageModeShared];
            id<MTLBuffer> antis = [engine->device newBufferWithLength:count options:MTLResourceStorageModeShared];
            if (!signs || !antis) {
                diagnostic(error, capacity, "Could not allocate Metal branch buffers.");
                goto done;
            }
            engine->signBuffer = signs;
            engine->antiBuffer = antis;
            engine->countCapacity = count;
        }
        {
            memcpy(engine->inputBuffer.contents, keys, keyBytes);
            id<MTLCommandBuffer> command = [engine->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!command || !encoder) {
                diagnostic(error, capacity, "Could not allocate the Metal command encoder.");
                goto done;
            }
            PauliParameters p = {(uint32_t)count, (uint32_t)(2 * n_words),
                                  (uint32_t)i, (uint32_t)j, qi, qj};
            [encoder setComputePipelineState:engine->pipeline];
            [encoder setBuffer:engine->inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:engine->outputBuffer offset:0 atIndex:1];
            [encoder setBuffer:engine->signBuffer offset:0 atIndex:2];
            [encoder setBuffer:engine->antiBuffer offset:0 atIndex:3];
            [encoder setBytes:&p length:sizeof(p) atIndex:4];
            NSUInteger width = engine->pipeline.maxTotalThreadsPerThreadgroup;
            if (width > 256) width = 256;
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                metal_diagnostic(error, capacity, "Metal command execution failed", command.error);
                goto done;
            }
            memcpy(branch_keys, engine->outputBuffer.contents, keyBytes);
            memcpy(branch_sign, engine->signBuffer.contents, count);
            memcpy(anticommute, engine->antiBuffer.contents, count);
            result = 0;
        }
    done:
        pthread_mutex_unlock(&engine->mutex);
        return result;
    }
}
