#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "checkpoint_rev.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    uint32_t h[8];
    uint64_t bytes;
    unsigned used;
    unsigned char block[64];
} sha256;
static uint32_t ror(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static void sha_block(sha256 *s, const unsigned char *p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) |
               ((uint32_t)p[4 * i + 2] << 8) | p[4 * i + 3];
    for (unsigned i = 16; i < 64; i++) {
        uint32_t a = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3),
                 b = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + a + w[i - 7] + b;
    }
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4], f = s->h[5],
             g = s->h[6], h = s->h[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ ((~e) & g)) + K[i] +
                      w[i],
                 t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    s->h[0] += a;
    s->h[1] += b;
    s->h[2] += c;
    s->h[3] += d;
    s->h[4] += e;
    s->h[5] += f;
    s->h[6] += g;
    s->h[7] += h;
}
static void sha_init(sha256 *s) {
    static const uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, h, sizeof(h));
    s->bytes = 0;
    s->used = 0;
}
static void sha_update(sha256 *s, const void *ptr, size_t n) {
    const unsigned char *p = ptr;
    s->bytes += n;
    while (n) {
        size_t take = 64 - s->used;
        if (take > n)
            take = n;
        memcpy(s->block + s->used, p, take);
        s->used += (unsigned)take;
        p += take;
        n -= take;
        if (s->used == 64) {
            sha_block(s, s->block);
            s->used = 0;
        }
    }
}
static void sha_final(sha256 *s, unsigned char out[32]) {
    uint64_t bits = s->bytes * 8;
    unsigned char pad[128] = {0x80};
    size_t n = s->used < 56 ? 56 - s->used : 120 - s->used;
    sha_update(s, pad, n);
    for (unsigned i = 0; i < 8; i++)
        pad[i] = (unsigned char)(bits >> (56 - 8 * i));
    sha_update(s, pad, 8);
    for (unsigned i = 0; i < 8; i++)
        for (unsigned j = 0; j < 4; j++)
            out[4 * i + j] = (unsigned char)(s->h[i] >> (24 - 8 * j));
}
void tpgp_payload_sha256(const void *p,size_t n,unsigned char out[32]){
    sha256 s;sha_init(&s);sha_update(&s,p,n);sha_final(&s,out);
}
static void error_set(char *e, size_t cap, const char *message) {
    if (e && cap)
        snprintf(e, cap, "%s", message);
}
static int same_double(double a, double b) {
    uint64_t x, y;
    memcpy(&x, &a, 8);
    memcpy(&y, &b, 8);
    return x == y;
}
static int product(size_t a, size_t b, size_t *c) {
    if (b && a > SIZE_MAX / b)
        return 0;
    *c = a * b;
    return 1;
}
static int canonical(const tpgp_checkpoint *s) {
    if (!s || FLT_RADIX != 2 || DBL_MANT_DIG != 53 || sizeof(double) != 8 || !s->map.n_spins ||
        s->map.n_spins > SIZE_MAX - 31 || s->map.n_words != (s->map.n_spins + 31) / 32 ||
        s->map.len > s->map.capacity || (!s->map.keys && s->map.len) ||
        (!s->map.values && s->map.len) || (!s->terms && s->term_count))
        return 0;
    if (!isfinite(s->dt) || s->dt <= 0 || s->step > s->total_steps || s->residual_depth < -1 ||
        s->residual_depth > 10 || s->initial_global < 0 || s->initial_global > 1 ||
        !memchr(s->model, 0, sizeof(s->model)) || !s->model[0] || !isfinite(s->j2) ||
        s->resumed_from_step > s->step || s->stop_reason < 0 || s->stop_reason > 6 ||
        s->last_status < TPGP_OK || s->last_status > TPGP_EBUDGET)
        return 0;
    size_t key_count, ignored;
    if (!product(s->map.len, s->map.n_words, &key_count) ||
        !product(key_count, sizeof(uint64_t), &ignored) ||
        !product(s->map.len, sizeof(double), &ignored) ||
        !product(s->term_count, sizeof(tpgp_term), &ignored) ||
        !isfinite(s->max_fid_bound) || s->max_fid_bound < 0 ||
        (s->max_fid_bound > 0 && !s->options.validated && s->residual_depth < 0) ||
        (s->stop_reason == 1 && s->step != s->total_steps))
        return 0;
    for (size_t k = 0; k < strlen(s->model); k++)
        if (!((s->model[k] >= 'a' && s->model[k] <= 'z') ||
              (s->model[k] >= '0' && s->model[k] <= '9') || s->model[k] == '_'))
            return 0;
    if (!isfinite(s->options.threshold) || s->options.threshold < 0 || !s->options.threads ||
        s->options.threads > 65536 || s->options.backend < 0 || s->options.backend > 4 ||
        s->options.validated < 0 || s->options.validated > 1 || !isfinite(s->options.max_drop_l2) ||
        s->options.max_drop_l2 < 0)
        return 0;
    if (s->options.max_terms && s->map.len > s->options.max_terms)
        return 0;
    const double ledger[] = {s->ledger.dropped_l1, s->ledger.dropped_l2,
                             s->ledger.dropped_l2_squared, s->ledger.arithmetic_l2, s->pf_bound};
    for (unsigned k = 0; k < 5; k++)
        if (isnan(ledger[k]) || ledger[k] < 0)
            return 0;
    if (s->ledger.arithmetic_enclosed < 0 || s->ledger.arithmetic_enclosed > 1 ||
        (s->ledger.arithmetic_enclosed && !s->options.validated) ||
        s->ledger.peak_terms < s->map.len)
        return 0;
    if (s->residual_depth >= 0 && (isnan(s->residual_bound) || s->residual_bound < 0))
        return 0;
    if (s->residual_depth < 0 && !isnan(s->residual_bound))
        return 0;
    for (size_t k = 0; k < s->map.len; k++) {
        if (!isfinite(s->map.values[k]) || s->map.values[k] == 0)
            return 0;
        size_t rem = s->map.n_spins % 32;
        if (rem && (s->map.keys[(k + 1) * s->map.n_words - 1] >> (2 * rem)))
            return 0;
        if (k) {
            int order = 0;
            for (size_t w = s->map.n_words; w > 0; w--) {
                uint64_t a = s->map.keys[(k - 1) * s->map.n_words + w - 1],
                         b = s->map.keys[k * s->map.n_words + w - 1];
                if (a < b) {
                    order = -1;
                    break;
                }
                if (a > b) {
                    order = 1;
                    break;
                }
            }
            if (order >= 0)
                return 0;
        }
    }
    for (size_t k = 0; k < s->term_count; k++) {
        const tpgp_term *t = s->terms + k;
        if (t->i >= s->map.n_spins || t->j >= s->map.n_spins || t->i == t->j || t->qi > 3 ||
            t->qj > 3 || !isfinite(t->h))
            return 0;
    }
    return 1;
}
void tpgp_checkpoint_free(tpgp_checkpoint *s) {
    if (s) {
        tpgp_map_free(&s->map);
        free(s->terms);
        memset(s, 0, sizeof(*s));
    }
}
typedef struct {
    FILE *f;
    sha256 hash;
    int ok;
} writer;
static void put(writer *w, const void *p, size_t n) {
    if (!w->ok)
        return;
    if (fwrite(p, 1, n, w->f) != n) {
        w->ok = 0;
        return;
    }
    sha_update(&w->hash, p, n);
}
static void put_u64(writer *w, uint64_t x) {
    unsigned char p[8];
    for (unsigned k = 0; k < 8; k++)
        p[k] = (unsigned char)(x >> (8 * k));
    put(w, p, 8);
}
static void put_double(writer *w, double x) {
    uint64_t u;
    memcpy(&u, &x, 8);
    put_u64(w, u);
}
static FILE *temporary(const char *path, char **name) {
    size_t n = strlen(path);
    if (n > SIZE_MAX - 16)
        return NULL;
    *name = malloc(n + 16);
    if (!*name)
        return NULL;
    snprintf(*name, n + 16, "%s.tmp.XXXXXX", path);
    int fd = mkstemp(*name);
    if (fd < 0) {
        free(*name);
        *name = NULL;
        return NULL;
    }
    FILE *f = fdopen(fd, "wb");
    if (!f) {
        close(fd);
        unlink(*name);
        free(*name);
        *name = NULL;
    }
    return f;
}
static int commit(FILE *f, const char *temporary_path, const char *path) {
    int ok = fflush(f) == 0 && !ferror(f);
    if (ok && fsync(fileno(f)))
        ok = 0;
    if (fclose(f))
        ok = 0;
    if (ok && rename(temporary_path, path))
        ok = 0;
    if (!ok) {
        unlink(temporary_path);
        return 0;
    }
    char *parent = strdup(path);
    if (!parent)
        return 0;
    char *slash = strrchr(parent, '/');
    if (!slash)
        strcpy(parent, ".");
    else if (slash == parent)
        slash[1] = 0;
    else
        *slash = 0;
    int fd = open(parent, O_RDONLY);
    free(parent);
    if (fd < 0)
        return 0;
    ok = fsync(fd) == 0;
    close(fd);
    return ok;
}
tpgp_status tpgp_checkpoint_save(const char *path, const tpgp_checkpoint *s, char *error,
                                 size_t cap) {
    if (!path || !*path || !canonical(s)) {
        error_set(error, cap, "invalid checkpoint state");
        return TPGP_EINVAL;
    }
    char *name = NULL;
    FILE *f = temporary(path, &name);
    if (!f) {
        error_set(error, cap, "cannot create checkpoint temporary file");
        return TPGP_EINVAL;
    }
    writer w = {f, {0}, 1};
    sha_init(&w.hash);
    put(&w, "TPGPRVCP", 8);
    put_u64(&w, 2);
    put_u64(&w, s->map.n_spins);
    put_u64(&w, s->map.len);
    put_u64(&w, s->term_count);
    put_u64(&w, s->step);
    put_u64(&w, s->total_steps);
    put_double(&w, s->dt);
    put_u64(&w, (uint64_t)(s->residual_depth + 1));
    put_u64(&w, (uint64_t)s->initial_global);
    put(&w, s->model, 32);
    put_double(&w, s->j2);
    put_u64(&w, s->linear_size);
    put_u64(&w, s->resumed_from_step);
    put_u64(&w, s->prior_total_steps);
    put_u64(&w, s->prior_max_terms);
    put_u64(&w, s->stop_reason);
    put_u64(&w, s->last_status);
    put_double(&w, s->options.threshold);
    put_u64(&w, s->options.max_terms);
    put_u64(&w, s->options.threads);
    put_u64(&w, s->options.backend);
    put_u64(&w, s->options.validated);
    put_double(&w, s->options.max_drop_l2);
    put_double(&w, s->ledger.dropped_l1);
    put_double(&w, s->ledger.dropped_l2);
    put_double(&w, s->ledger.dropped_l2_squared);
    put_double(&w, s->ledger.arithmetic_l2);
    put_u64(&w, s->ledger.rotations);
    put_u64(&w, s->ledger.removal_events);
    put_u64(&w, s->ledger.peak_terms);
    put_u64(&w, s->ledger.arithmetic_enclosed);
    put_double(&w, s->pf_bound);
    put_double(&w, s->residual_bound);
    put_double(&w, s->max_fid_bound);
    for (size_t k = 0; k < s->term_count; k++) {
        const tpgp_term *t = s->terms + k;
        put_u64(&w, t->i);
        put_u64(&w, t->j);
        put_u64(&w, t->qi);
        put_u64(&w, t->qj);
        put_double(&w, t->h);
        put_u64(&w, t->layer);
    }
    for (size_t k = 0; k < s->map.len; k++) {
        for (size_t j = 0; j < s->map.n_words; j++)
            put_u64(&w, s->map.keys[k * s->map.n_words + j]);
        put_double(&w, s->map.values[k]);
    }
    unsigned char digest[32];
    sha_final(&w.hash, digest);
    if (fwrite(digest, 1, 32, f) != 32)
        w.ok = 0;
    if (!w.ok) {
        fclose(f);
        unlink(name);
        free(name);
        error_set(error, cap, "checkpoint write failed; old record preserved");
        return TPGP_EINVAL;
    }
    if (!commit(f, name, path)) {
        free(name);
        error_set(error, cap, "checkpoint atomic commit or directory synchronization failed");
        return TPGP_EINVAL;
    }
    free(name);
    size_t n = strlen(path);
    char *manifest = malloc(n + 20);
    if (!manifest) {
        error_set(error, cap, "checkpoint saved but manifest allocation failed");
        return TPGP_ENOMEM;
    }
    snprintf(manifest, n + 20, "%s.manifest.json", path);
    f = temporary(manifest, &name);
    if (!f) {
        free(manifest);
        error_set(error, cap, "checkpoint saved but manifest creation failed");
        return TPGP_EINVAL;
    }
    char hex[65];
    for (unsigned k = 0; k < 32; k++)
        snprintf(hex + 2 * k, 3, "%02x", digest[k]);
    fprintf(f,
            "{\n  \"format\": \"TPGPRVCP-2\",\n  \"sha256_payload\": \"%s\",\n  \"model\": "
            "\"%s\",\n  \"n_spins\": %zu,\n  \"active_terms\": %zu,\n  \"hamiltonian_terms\": "
            "%zu,\n  \"step\": %zu,\n  \"total_steps\": %zu,\n  \"dt_hex\": \"%a\",\n  "
            "\"backend\": %d,\n  \"threads\": %u,\n  \"validated\": %d,\n  \"threshold_hex\": "
            "\"%a\",\n  \"max_terms\": %zu,\n  \"residual_depth\": %d,\n",
            hex, s->model, s->map.n_spins, s->map.len, s->term_count, s->step, s->total_steps,
            s->dt, s->options.backend, s->options.threads, s->options.validated,
            s->options.threshold, s->options.max_terms, s->residual_depth);
    fprintf(f, "  \"max_fid_bound_hex\": \"%a\",\n", s->max_fid_bound);
    static const char *reasons[] = {"running", "complete", "walltime", "signal",
                                   "resource_failure", "numerical_failure", "accuracy_limit"};
    fprintf(f,
            "  \"resumed_from_step\": %zu,\n  \"prior_total_steps\": %zu,\n  \"prior_max_terms\": "
            "%zu,\n  \"stop_reason\": \"%s\",\n  \"status_code\": %d,\n  \"status\": \"%s\"\n}\n",
            s->resumed_from_step, s->prior_total_steps, s->prior_max_terms, reasons[s->stop_reason],
            s->last_status, tpgp_status_string(s->last_status));
    int ok = commit(f, name, manifest);
    free(name);
    free(manifest);
    if (!ok) {
        error_set(error, cap, "checkpoint saved but manifest commit failed");
        return TPGP_EINVAL;
    }
    error_set(error, cap, "ok");
    return TPGP_OK;
}
typedef struct {
    FILE *f;
    uint64_t left;
    int ok;
    sha256 hash;
} reader;
static void get(reader *r, void *p, size_t n) {
    if (!r->ok || n > r->left || fread(p, 1, n, r->f) != n) {
        r->ok = 0;
        memset(p, 0, n);
        return;
    }
    r->left -= n;
    sha_update(&r->hash, p, n);
}
static uint64_t get_u64(reader *r) {
    unsigned char p[8];
    get(r, p, 8);
    uint64_t x = 0;
    for (unsigned k = 0; k < 8; k++)
        x |= (uint64_t)p[k] << (8 * k);
    return x;
}
static size_t get_size(reader *r) {
    uint64_t x = get_u64(r);
    if (x > SIZE_MAX)
        r->ok = 0;
    return (size_t)x;
}
static double get_double(reader *r) {
    uint64_t x = get_u64(r);
    double d;
    memcpy(&d, &x, 8);
    return d;
}
tpgp_status tpgp_checkpoint_load(const char *path, tpgp_checkpoint *dst, char *error, size_t cap) {
    if (!path || !dst || FLT_RADIX != 2 || DBL_MANT_DIG != 53 || sizeof(double) != 8) {
        error_set(error, cap, "invalid checkpoint load arguments");
        return TPGP_EINVAL;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        error_set(error, cap, "cannot open checkpoint");
        return TPGP_EINVAL;
    }
    struct stat st;
    if (fstat(fileno(f), &st) || st.st_size < 328 || (uintmax_t)st.st_size > UINT64_MAX / 8) {
        fclose(f);
        error_set(error, cap, "checkpoint is truncated or has invalid length");
        return TPGP_EINVAL;
    }
    uint64_t payload = (uint64_t)st.st_size - 32, left = payload;
    unsigned char buf[65536], expected[32], actual[32];
    sha256 hash;
    sha_init(&hash);
    int good = 1;
    while (left) {
        size_t n = left > sizeof(buf) ? sizeof(buf) : (size_t)left;
        if (fread(buf, 1, n, f) != n) {
            good = 0;
            break;
        }
        sha_update(&hash, buf, n);
        left -= n;
    }
    sha_final(&hash, actual);
    if (!good || fread(expected, 1, 32, f) != 32 || memcmp(actual, expected, 32) ||
        fgetc(f) != EOF || ferror(f)) {
        fclose(f);
        error_set(error, cap, "checkpoint checksum/length mismatch");
        return TPGP_EINVAL;
    }
    rewind(f);
    reader r = {f, payload, 1, {0}};
    sha_init(&r.hash);
    char magic[8];
    get(&r, magic, 8);
    uint64_t version = get_u64(&r);
    tpgp_checkpoint s = {0};
    size_t spins = get_size(&r), count = get_size(&r);
    s.term_count = get_size(&r);
    s.step = get_size(&r);
    s.total_steps = get_size(&r);
    s.dt = get_double(&r);
    uint64_t depth = get_u64(&r), global = get_u64(&r);
    if (depth > 11 || global > 1)
        r.ok = 0;
    s.residual_depth = (int)depth - 1;
    s.initial_global = (int)global;
    get(&r, s.model, 32);
    s.j2 = get_double(&r);
    s.linear_size = get_size(&r);
    s.resumed_from_step = get_size(&r);
    s.prior_total_steps = get_size(&r);
    s.prior_max_terms = get_size(&r);
    uint64_t reason = get_u64(&r), last_status = get_u64(&r);
    if (reason > 6 || last_status > TPGP_EBUDGET)
        r.ok = 0;
    s.stop_reason = (int)reason;
    s.last_status = (tpgp_status)last_status;
    s.options = tpgp_default_options();
    s.options.threshold = get_double(&r);
    s.options.max_terms = get_size(&r);
    uint64_t threads = get_u64(&r), backend = get_u64(&r), validated = get_u64(&r);
    if (threads > 65536 || backend > 4 || validated > 1)
        r.ok = 0;
    s.options.threads = (unsigned)threads;
    s.options.backend = (tpgp_backend)backend;
    s.options.validated = (int)validated;
    s.options.max_drop_l2 = get_double(&r);
    s.ledger.dropped_l1 = get_double(&r);
    s.ledger.dropped_l2 = get_double(&r);
    s.ledger.dropped_l2_squared = get_double(&r);
    s.ledger.arithmetic_l2 = get_double(&r);
    s.ledger.rotations = get_size(&r);
    s.ledger.removal_events = get_size(&r);
    s.ledger.peak_terms = get_size(&r);
    uint64_t enclosed = get_u64(&r);
    if (enclosed > 1)
        r.ok = 0;
    s.ledger.arithmetic_enclosed = (int)enclosed;
    s.pf_bound = get_double(&r);
    s.residual_bound = get_double(&r);
    s.max_fid_bound = get_double(&r);
    size_t words = spins <= SIZE_MAX - 31 ? (spins + 31) / 32 : 0, keys = 0, keybytes = 0,
           termbytes = 0, diskterms = 0, diskmap = 0;
    if (!r.ok || memcmp(magic, "TPGPRVCP", 8) || version != 2 || !spins || !words ||
        !product(count, words, &keys) || !product(keys, 8, &keybytes) ||
        !product(s.term_count, sizeof(tpgp_term), &termbytes) ||
        !product(s.term_count, 48, &diskterms) || words == SIZE_MAX ||
        !product(count, words + 1, &diskmap) || !product(diskmap, 8, &diskmap) ||
        diskterms > UINT64_MAX - diskmap || r.left != (uint64_t)diskterms + diskmap ||
        count > SIZE_MAX / sizeof(double))
        goto invalid;
    tpgp_status status = tpgp_map_init(&s.map, spins);
    if (status)
        goto invalid;
    s.map.len = s.map.capacity = count;
    s.map.keys = count ? malloc(keybytes) : NULL;
    s.map.values = count ? malloc(count * sizeof(double)) : NULL;
    s.terms = s.term_count ? malloc(termbytes) : NULL;
    if ((count && (!s.map.keys || !s.map.values)) || (s.term_count && !s.terms)) {
        tpgp_checkpoint_free(&s);
        fclose(f);
        error_set(error, cap, "checkpoint allocation failed");
        return TPGP_ENOMEM;
    }
    for (size_t k = 0; k < s.term_count; k++) {
        tpgp_term *t = s.terms + k;
        t->i = get_size(&r);
        t->j = get_size(&r);
        uint64_t qi = get_u64(&r), qj = get_u64(&r);
        if (qi > 3 || qj > 3)
            r.ok = 0;
        t->qi = (unsigned)qi;
        t->qj = (unsigned)qj;
        t->h = get_double(&r);
        t->layer = get_size(&r);
    }
    for (size_t k = 0; k < count; k++) {
        for (size_t j = 0; j < words; j++)
            s.map.keys[k * words + j] = get_u64(&r);
        s.map.values[k] = get_double(&r);
    }
    unsigned char parsed_hash[32];
    sha_final(&r.hash, parsed_hash);
    if (!r.ok || r.left || memcmp(parsed_hash, expected, 32) || !canonical(&s))
        goto invalid;
    fclose(f);
    tpgp_checkpoint_free(dst);
    *dst = s;
    error_set(error, cap, "ok");
    return TPGP_OK;
invalid:
    tpgp_checkpoint_free(&s);
    fclose(f);
    error_set(error, cap, "checkpoint format or numerical invariants are invalid");
    return TPGP_EINVAL;
}
int tpgp_checkpoint_same_configuration(const tpgp_checkpoint *a, const tpgp_checkpoint *b) {
    if (!a || !b || a->map.n_spins != b->map.n_spins || a->term_count != b->term_count ||
        a->total_steps != b->total_steps || !same_double(a->dt, b->dt) ||
        a->residual_depth != b->residual_depth || a->initial_global != b->initial_global ||
        strncmp(a->model, b->model, 32) || !same_double(a->j2, b->j2) ||
        a->linear_size != b->linear_size)
        return 0;
    const tpgp_options *x = &a->options, *y = &b->options;
    if (!same_double(x->threshold, y->threshold) || x->max_terms != y->max_terms ||
        x->threads != y->threads || x->backend != y->backend || x->validated != y->validated ||
        !same_double(x->max_drop_l2, y->max_drop_l2))
        return 0;
    for (size_t k = 0; k < a->term_count; k++) {
        const tpgp_term *x = a->terms + k, *y = b->terms + k;
        if (x->i != y->i || x->j != y->j || x->qi != y->qi || x->qj != y->qj ||
            !same_double(x->h, y->h) || x->layer != y->layer)
            return 0;
    }
    return 1;
}
