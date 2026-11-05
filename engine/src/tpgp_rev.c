#include "tpgp_rev.h"
#include "tpgp_cuda_rev.h"
#include "tpgp_metal_rev.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#ifdef TPGP_HAVE_MPFR
#include <mpfr.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* No fast-math: the arithmetic enclosure assumes IEEE-754 binary64,
   round-to-nearest, gradual underflow and separately rounded operations. */
static int size_product(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b)
        return 0;
    *out = a * b;
    return 1;
}
static double upadd(double a, double b) {
    if (a == 0)
        return b;
    if (b == 0)
        return a;
    return nextafter(a + b, INFINITY);
}
static double upmul(double a, double b) {
    if (a == 0 || b == 0)
        return 0;
    return nextafter(a * b, INFINITY);
}
static double round_radius(double x) {
    if (!isfinite(x))
        return INFINITY;
    double a = fabs(nextafter(x, INFINITY) - x), b = fabs(x - nextafter(x, -INFINITY));
    return fmax(a, b);
}
static double upper_l2(const double *v, size_t n) {
#ifdef TPGP_HAVE_MPFR
    mpfr_t sum, x;
    mpfr_inits2(192, sum, x, (mpfr_ptr)0);
    mpfr_set_zero(sum, 0);
    for (size_t k = 0; k < n; k++) {
        mpfr_set_d(x, v[k], MPFR_RNDN);
        mpfr_sqr(x, x, MPFR_RNDU);
        mpfr_add(sum, sum, x, MPFR_RNDU);
    }
    mpfr_sqrt(sum, sum, MPFR_RNDU);
    double out = mpfr_get_d(sum, MPFR_RNDU);
    mpfr_clears(sum, x, (mpfr_ptr)0);
    return out;
#else
    double sum = 0;
    for (size_t k = 0; k < n; k++)
        sum = upadd(sum, upmul(v[k], v[k]));
    return sum == 0 ? 0 : nextafter(sqrt(sum), INFINITY);
#endif
}
static int keycmp(const uint64_t *a, const uint64_t *b, size_t words) {
    for (size_t w = words; w > 0; w--) {
        if (a[w - 1] < b[w - 1])
            return -1;
        if (a[w - 1] > b[w - 1])
            return 1;
    }
    return 0;
}
static int key_valid(const tpgp_map *m, const uint64_t *key) {
    size_t rem = m->n_spins % 32;
    return !rem || !(key[m->n_words - 1] >> (2 * rem));
}
static int map_layout_valid(const tpgp_map *m) {
    size_t bytes, count;
    return m && m->n_spins && m->n_spins <= SIZE_MAX - 31 && m->n_words == (m->n_spins + 31) / 32 &&
           m->len <= m->capacity && size_product(m->capacity, m->n_words, &count) &&
           size_product(count, sizeof(uint64_t), &bytes) &&
           m->capacity <= SIZE_MAX / sizeof(double) && (!m->capacity || (m->keys && m->values));
}
static tpgp_status reserve(tpgp_map *m, size_t n) {
    if (n <= m->capacity)
        return TPGP_OK;
    size_t cap = m->capacity ? m->capacity : 8;
    while (cap < n) {
        if (cap > SIZE_MAX / 2) {
            cap = n;
            break;
        }
        cap *= 2;
    }
    size_t nk, bytes;
    if (!size_product(cap, m->n_words, &nk) || !size_product(nk, sizeof(uint64_t), &bytes) ||
        cap > SIZE_MAX / sizeof(double))
        return TPGP_ENOMEM;
    uint64_t *keys = malloc(bytes);
    double *values = malloc(cap * sizeof(double));
    if (!keys || !values) {
        free(keys);
        free(values);
        return TPGP_ENOMEM;
    }
    if (m->len) {
        memcpy(keys, m->keys, m->len * m->n_words * sizeof(uint64_t));
        memcpy(values, m->values, m->len * sizeof(double));
    }
    free(m->keys);
    free(m->values);
    m->keys = keys;
    m->values = values;
    m->capacity = cap;
    return TPGP_OK;
}
const char *tpgp_status_string(tpgp_status s) {
    static const char *names[] = {"ok",
                                  "invalid argument",
                                  "allocation failed",
                                  "active-term cap exceeded",
                                  "nonfinite arithmetic or unsupported floating-point environment",
                                  "backend unavailable",
                                  "coefficient-removal budget exceeded"};
    return s >= TPGP_OK && s <= TPGP_EBUDGET ? names[s] : "unknown status";
}
int tpgp_has_mpfr(void) {
#ifdef TPGP_HAVE_MPFR
    return 1;
#else
    return 0;
#endif
}
int tpgp_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
tpgp_options tpgp_default_options(void) {
    tpgp_options o = {0, 0, 1, TPGP_SERIAL, 0, 0, NULL, NULL};
    return o;
}
void tpgp_ledger_init(tpgp_ledger *l, int validated) {
    memset(l, 0, sizeof(*l));
    l->arithmetic_enclosed = validated && tpgp_has_mpfr();
}
tpgp_status tpgp_map_init(tpgp_map *m, size_t n) {
    if (!m || !n || n > SIZE_MAX - 31)
        return TPGP_EINVAL;
    memset(m, 0, sizeof(*m));
    m->n_spins = n;
    m->n_words = (n + 31) / 32;
    return TPGP_OK;
}
void tpgp_map_free(tpgp_map *m) {
    if (m) {
        free(m->keys);
        free(m->values);
        memset(m, 0, sizeof(*m));
    }
}
tpgp_status tpgp_map_clone(tpgp_map *dst, const tpgp_map *src) {
    if (!dst || !map_layout_valid(src))
        return TPGP_EINVAL;
    if (dst == src)
        return TPGP_OK;
    tpgp_map tmp;
    tpgp_status s = tpgp_map_init(&tmp, src->n_spins);
    if (s)
        return s;
    s = reserve(&tmp, src->len);
    if (s) {
        tpgp_map_free(&tmp);
        return s;
    }
    tmp.len = src->len;
    if (src->len) {
        memcpy(tmp.keys, src->keys, src->len * src->n_words * sizeof(uint64_t));
        memcpy(tmp.values, src->values, src->len * sizeof(double));
    }
    tpgp_map_free(dst);
    *dst = tmp;
    return TPGP_OK;
}
unsigned tpgp_key_get(const uint64_t *key, size_t site) {
    return (unsigned)((key[site / 32] >> (2 * (site % 32))) & 3u);
}
void tpgp_key_set(uint64_t *key, size_t site, unsigned p) {
    size_t w = site / 32, s = 2 * (site % 32);
    key[w] = (key[w] & ~(UINT64_C(3) << s)) | ((uint64_t)(p & 3u) << s);
}
static size_t lower_bound(const tpgp_map *m, const uint64_t *key, int *found) {
    size_t lo = 0, hi = m->len;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (keycmp(m->keys + mid * m->n_words, key, m->n_words) < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    *found = lo < m->len && keycmp(m->keys + lo * m->n_words, key, m->n_words) == 0;
    return lo;
}
tpgp_status tpgp_map_add(tpgp_map *m, const uint64_t *key, double v) {
    if (!map_layout_valid(m) || !key || !isfinite(v) || !key_valid(m, key))
        return TPGP_EINVAL;
    if (v == 0)
        return TPGP_OK;
    int found;
    size_t pos = lower_bound(m, key, &found);
    if (found) {
        double sum = m->values[pos] + v;
        if (!isfinite(sum))
            return TPGP_ENUMERIC;
        if (sum == 0) {
            memmove(m->keys + pos * m->n_words, m->keys + (pos + 1) * m->n_words,
                    (m->len - pos - 1) * m->n_words * sizeof(uint64_t));
            memmove(m->values + pos, m->values + pos + 1, (m->len - pos - 1) * sizeof(double));
            m->len--;
        } else
            m->values[pos] = sum;
        return TPGP_OK;
    }
    uint64_t *copy = malloc(m->n_words * sizeof(uint64_t));
    if (!copy)
        return TPGP_ENOMEM;
    memcpy(copy, key, m->n_words * sizeof(uint64_t));
    tpgp_status s = reserve(m, m->len + 1);
    if (s) {
        free(copy);
        return s;
    }
    memmove(m->keys + (pos + 1) * m->n_words, m->keys + pos * m->n_words,
            (m->len - pos) * m->n_words * sizeof(uint64_t));
    memmove(m->values + pos + 1, m->values + pos, (m->len - pos) * sizeof(double));
    memcpy(m->keys + pos * m->n_words, copy, m->n_words * sizeof(uint64_t));
    free(copy);
    m->values[pos] = v;
    m->len++;
    return TPGP_OK;
}
tpgp_status tpgp_map_add_single(tpgp_map *m, size_t site, unsigned p, double v) {
    if (!m || site >= m->n_spins || p > 3)
        return TPGP_EINVAL;
    uint64_t *key = calloc(m->n_words, sizeof(uint64_t));
    if (!key)
        return TPGP_ENOMEM;
    tpgp_key_set(key, site, p);
    tpgp_status s = tpgp_map_add(m, key, v);
    free(key);
    return s;
}
double tpgp_map_get(const tpgp_map *m, const uint64_t *key) {
    int found;
    size_t p = lower_bound(m, key, &found);
    return found ? m->values[p] : 0;
}
double tpgp_map_l1(const tpgp_map *m) {
    long double s = 0;
    for (size_t k = 0; k < m->len; k++)
        s += fabsl(m->values[k]);
    return (double)s;
}
double tpgp_map_l2(const tpgp_map *m) {
    long double s = 0;
    for (size_t k = 0; k < m->len; k++)
        s += (long double)m->values[k] * m->values[k];
    return (double)sqrtl(s);
}
double tpgp_map_distance(const tpgp_map *a, const tpgp_map *b) {
    if (a->n_spins != b->n_spins)
        return NAN;
    size_t i = 0, j = 0;
    long double sum = 0;
    while (i < a->len || j < b->len) {
        int c = i == a->len ? 1
                : j == b->len
                    ? -1
                    : keycmp(a->keys + i * a->n_words, b->keys + j * b->n_words, a->n_words);
        long double v;
        if (c < 0)
            v = a->values[i++];
        else if (c > 0)
            v = -b->values[j++];
        else {
            v = (long double)a->values[i++] - b->values[j++];
        }
        sum += v * v;
    }
    return (double)sqrtl(sum);
}
double tpgp_map_dot(const tpgp_map *a, const tpgp_map *b) {
    if (a->n_spins != b->n_spins)
        return NAN;
    size_t i = 0, j = 0;
    long double sum = 0;
    while (i < a->len && j < b->len) {
        int c = keycmp(a->keys + i * a->n_words, b->keys + j * b->n_words, a->n_words);
        if (c < 0)
            i++;
        else if (c > 0)
            j++;
        else {
            sum += (long double)a->values[i++] * b->values[j++];
        }
    }
    return (double)sum;
}
double tpgp_map_single_sum(const tpgp_map *m, unsigned p) {
    if (p < 1 || p > 3)
        return NAN;
    long double sum = 0, correction = 0;
    for (size_t k = 0; k < m->len; k++) {
        unsigned weight = 0, symbol = 0;
        for (size_t w = 0; w < m->n_words; w++) {
            uint64_t q = m->keys[k * m->n_words + w];
            while (q) {
                unsigned x = (unsigned)(q & 3u);
                if (x) {
                    weight++;
                    symbol = x;
                }
                q >>= 2;
            }
        }
        if (weight == 1 && symbol == p) {
            long double value = m->values[k], next = sum + value;
            if (fabsl(sum) >= fabsl(value))
                correction += (sum - next) + value;
            else
                correction += (value - next) + sum;
            sum = next;
        }
    }
    return (double)(sum + correction);
}
tpgp_status tpgp_readout(const tpgp_map *m, unsigned p, size_t normalization, double *midpoint,
                         double *radius) {
    if (!map_layout_valid(m) || !normalization || p < 1 || p > 3 || !midpoint || !radius)
        return TPGP_EINVAL;
#ifdef TPGP_HAVE_MPFR
    mpfr_t lo, hi, mid, v, den, r;
    mpfr_inits2(192, lo, hi, mid, v, den, r, (mpfr_ptr)0);
    mpfr_set_zero(lo, 0);
    mpfr_set_zero(hi, 0);
    mpfr_set_zero(mid, 0);
    mpfr_set_uj(den, (uintmax_t)normalization, MPFR_RNDN);
    for (size_t k = 0; k < m->len; k++) {
        unsigned weight = 0, symbol = 0;
        for (size_t w = 0; w < m->n_words; w++) {
            uint64_t key = m->keys[k * m->n_words + w];
            while (key) {
                unsigned x = (unsigned)(key & 3u);
                if (x) {
                    weight++;
                    symbol = x;
                }
                key >>= 2;
            }
        }
        if (weight != 1 || symbol != p)
            continue;
        mpfr_set_d(v, m->values[k], MPFR_RNDN);
        mpfr_add(lo, lo, v, MPFR_RNDD);
        mpfr_add(hi, hi, v, MPFR_RNDU);
        mpfr_add(mid, mid, v, MPFR_RNDN);
    }
    mpfr_div(lo, lo, den, MPFR_RNDD);
    mpfr_div(hi, hi, den, MPFR_RNDU);
    mpfr_div(mid, mid, den, MPFR_RNDN);
    double center = mpfr_get_d(mid, MPFR_RNDN);
    mpfr_set_d(mid, center, MPFR_RNDN);
    mpfr_sub(r, mid, lo, MPFR_RNDU);
    mpfr_sub(v, hi, mid, MPFR_RNDU);
    mpfr_max(r, r, v, MPFR_RNDU);
    double rad = mpfr_get_d(r, MPFR_RNDU);
    mpfr_clears(lo, hi, mid, v, den, r, (mpfr_ptr)0);
    if (!isfinite(center) || !isfinite(rad))
        return TPGP_ENUMERIC;
    *midpoint = center;
    *radius = rad;
    return TPGP_OK;
#else
    return TPGP_EBACKEND;
#endif
}
tpgp_status tpgp_map_add_scaled(tpgp_map *dst, const tpgp_map *a, const tpgp_map *b, double scale) {
    if (!dst || !a || !b || a->n_spins != b->n_spins || !isfinite(scale))
        return TPGP_EINVAL;
    tpgp_map tmp = {0};
    tpgp_status s = tpgp_map_clone(&tmp, a);
    if (s)
        return s;
    for (size_t k = 0; k < b->len; k++) {
        double v = scale * b->values[k];
        if (!isfinite(v)) {
            s = TPGP_ENUMERIC;
            break;
        }
        s = tpgp_map_add(&tmp, b->keys + k * b->n_words, v);
        if (s)
            break;
    }
    if (s) {
        tpgp_map_free(&tmp);
        return s;
    }
    tpgp_map_free(dst);
    *dst = tmp;
    return TPGP_OK;
}
unsigned tpgp_pauli_phase(unsigned q, unsigned p) {
    static const unsigned phase[4][4] = {{0, 0, 0, 0}, {0, 0, 1, 3}, {0, 3, 0, 1}, {0, 1, 3, 0}};
    return q < 4 && p < 4 ? phase[q][p] : 4;
}
static int anticomm(unsigned q, unsigned p) { return q && p && q != p; }

typedef struct {
    const tpgp_map *in;
    tpgp_map *raw;
    double *radii;
    size_t start, end, i, j;
    unsigned qi, qj;
    double c, s, rc, rs;
    int enclose;
    const uint64_t *metal_keys;
    const int8_t *metal_sign;
    const uint8_t *metal_anti;
} branch_job;
static void *branches(void *ptr) {
    branch_job *b = ptr;
    size_t words = b->in->n_words;
    for (size_t k = b->start; k < b->end; k++) {
        const uint64_t *key = b->in->keys + k * words;
        uint64_t *a = b->raw->keys + (2 * k) * words, *d = a + words;
        memcpy(a, key, words * sizeof(uint64_t));
        memcpy(d, key, words * sizeof(uint64_t));
        double v = b->in->values[k];
        unsigned pi = tpgp_key_get(key, b->i), pj = tpgp_key_get(key, b->j);
        int anti = b->metal_anti ? b->metal_anti[k] : (anticomm(b->qi, pi) ^ anticomm(b->qj, pj));
        if (!anti) {
            b->raw->values[2 * k] = v;
            b->raw->values[2 * k + 1] = 0;
            continue;
        }
        int sign;
        if (b->metal_keys) {
            memcpy(d, b->metal_keys + k * words, words * sizeof(uint64_t));
            sign = b->metal_sign[k];
        } else {
            unsigned phase = (tpgp_pauli_phase(b->qi, pi) + tpgp_pauli_phase(b->qj, pj)) & 3u;
            sign = phase == 1 ? 1 : -1;
            tpgp_key_set(d, b->i, pi ^ b->qi);
            tpgp_key_set(d, b->j, pj ^ b->qj);
        }
        double x = v * b->c, y = v * b->s;
        b->raw->values[2 * k] = x;
        b->raw->values[2 * k + 1] = sign * y;
        if (b->enclose) {
            b->radii[2 * k] = upadd(upmul(fabs(v), b->rc), round_radius(x));
            b->radii[2 * k + 1] = upadd(upmul(fabs(v), b->rs), round_radius(y));
        }
    }
    return NULL;
}
static void stable_sort_indices(size_t *idx, size_t *tmp, size_t lo, size_t hi, const tpgp_map *m) {
    if (hi - lo < 2)
        return;
    size_t mid = lo + (hi - lo) / 2;
    stable_sort_indices(idx, tmp, lo, mid, m);
    stable_sort_indices(idx, tmp, mid, hi, m);
    size_t i = lo, j = mid, k = lo;
    while (i < mid || j < hi) {
        if (j == hi || (i < mid && keycmp(m->keys + idx[i] * m->n_words,
                                          m->keys + idx[j] * m->n_words, m->n_words) <= 0))
            tmp[k++] = idx[i++];
        else
            tmp[k++] = idx[j++];
    }
    memcpy(idx + lo, tmp + lo, (hi - lo) * sizeof(size_t));
}
static tpgp_status trigonometry(double h, double duration, int validated, double *c, double *s,
                                double *rc, double *rs) {
    *rc = *rs = 0;
#ifdef TPGP_HAVE_MPFR
    if (validated) {
        mpfr_t x, y, lo, hi, mid, rad;
        mpfr_inits2(256, x, y, lo, hi, mid, rad, (mpfr_ptr)0);
        mpfr_set_d(x, h, MPFR_RNDN);
        mpfr_set_d(y, duration, MPFR_RNDN);
        mpfr_mul(x, x, y, MPFR_RNDN);
        mpfr_mul_2ui(x, x, 1, MPFR_RNDN);
        /* The exact product of two finite binary64 numbers needs <=106 bits;
           MPFR's unbounded exponent range prevents binary64 underflow here. */
        for (int k = 0; k < 2; k++) {
            if (k) {
                mpfr_sin(lo, x, MPFR_RNDD);
                mpfr_sin(hi, x, MPFR_RNDU);
                mpfr_sin(mid, x, MPFR_RNDN);
            } else {
                mpfr_cos(lo, x, MPFR_RNDD);
                mpfr_cos(hi, x, MPFR_RNDU);
                mpfr_cos(mid, x, MPFR_RNDN);
            }
            double m = mpfr_get_d(mid, MPFR_RNDN);
            mpfr_set_d(mid, m, MPFR_RNDN);
            mpfr_sub(rad, mid, lo, MPFR_RNDU);
            mpfr_sub(y, hi, mid, MPFR_RNDU);
            mpfr_max(rad, rad, y, MPFR_RNDU);
            double r = mpfr_get_d(rad, MPFR_RNDU);
            if (k) {
                *s = m;
                *rs = r;
            } else {
                *c = m;
                *rc = r;
            }
        }
        mpfr_clears(x, y, lo, hi, mid, rad, (mpfr_ptr)0);
        return TPGP_OK;
    }
#else
    if (validated)
        return TPGP_EBACKEND;
#endif
    double x = 2 * h * duration;
    if (!isfinite(x))
        return TPGP_ENUMERIC;
    *c = cos(x);
    *s = sin(x);
    return TPGP_OK;
}
static tpgp_status rotate_product(tpgp_map *map, size_t i, size_t j, unsigned qi, unsigned qj,
                                  double h, double duration, const tpgp_options *o,
                                  tpgp_ledger *ledger) {
    if (!map_layout_valid(map) || !o || !ledger || i >= map->n_spins || j >= map->n_spins ||
        i == j || qi > 3 || qj > 3 || !isfinite(h) || !isfinite(duration) ||
        !isfinite(o->threshold) || o->threshold < 0 || !isfinite(o->max_drop_l2) ||
        o->max_drop_l2 < 0 || o->threads == 0)
        return TPGP_EINVAL;
    if (!isfinite(ledger->dropped_l1) || ledger->dropped_l1 < 0 || !isfinite(ledger->dropped_l2) ||
        ledger->dropped_l2 < 0 || !isfinite(ledger->dropped_l2_squared) ||
        ledger->dropped_l2_squared < 0 || !isfinite(ledger->arithmetic_l2) ||
        ledger->arithmetic_l2 < 0)
        return TPGP_EINVAL;
    if (o->validated) {
        volatile double small = DBL_MIN, factor = .5;
        if (FLT_RADIX != 2 || DBL_MANT_DIG != 53 || fegetround() != FE_TONEAREST ||
            small * factor == 0)
            return TPGP_ENUMERIC;
    }
    if (o->backend < TPGP_SERIAL || o->backend > TPGP_CUDA)
        return TPGP_EBACKEND;
    if (o->backend == TPGP_OPENMP && !tpgp_has_openmp())
        return TPGP_EBACKEND;
    double c, s, rc, rs;
    tpgp_status status = trigonometry(h, duration, o->validated, &c, &s, &rc, &rs);
    if (status)
        return status;
    if (h == 0 || duration == 0 || map->len == 0)
        return TPGP_OK;
    if (map->len > SIZE_MAX / 2)
        return TPGP_ENOMEM;
    tpgp_map raw = {0}, out = {0};
    tpgp_map_init(&raw, map->n_spins);
    tpgp_map_init(&out, map->n_spins);
    size_t count = 2 * map->len;
    status = reserve(&raw, count);
    if (status)
        goto done;
    raw.len = count;
    double *rad = NULL, *removed = NULL, *merged_rad = NULL;
    size_t *idx = NULL, *tmp = NULL;
    uint64_t *mk = NULL;
    int8_t *ms = NULL;
    uint8_t *ma = NULL;
    rad = calloc(count, sizeof(double));
    idx = malloc(count * sizeof(size_t));
    tmp = malloc(count * sizeof(size_t));
    removed = malloc(count * sizeof(double));
    merged_rad = malloc(count * sizeof(double));
    if (!rad || !idx || !tmp || !removed || !merged_rad) {
        status = TPGP_ENOMEM;
        goto buffers;
    }
    if (o->backend == TPGP_METAL) {
        size_t bytes;
        if (!size_product(map->len * map->n_words, sizeof(uint64_t), &bytes)) {
            status = TPGP_ENOMEM;
            goto buffers;
        }
        mk = malloc(bytes);
        ms = malloc(map->len);
        ma = malloc(map->len);
        if (!mk || !ms || !ma) {
            status = TPGP_ENOMEM;
            goto buffers;
        }
        char err[256];
        tpgp_metal_context *ctx =
            o->metal_context ? o->metal_context : tpgp_metal_create(err, sizeof(err));
        if (!ctx) {
            status = TPGP_EBACKEND;
            goto buffers;
        }
        int result = tpgp_metal_transform(ctx, map->keys, map->len, map->n_words, map->n_spins, i,
                                          j, qi, qj, mk, ms, ma, err, sizeof(err));
        if (!o->metal_context)
            tpgp_metal_destroy(ctx);
        if (result != 0) {
            status = TPGP_EBACKEND;
            goto buffers;
        }
    }
    branch_job base = {map, &raw, rad, 0,  map->len,     i,  j,  qi, qj,
                       c,   s,    rc,  rs, o->validated, mk, ms, ma};
    unsigned nt = o->threads;
    if (nt > map->len)
        nt = (unsigned)map->len;
    if (o->backend == TPGP_CUDA) {
        char error[256];
        tpgp_cuda_context *context =
            o->cuda_context ? o->cuda_context : tpgp_cuda_create(0, error, sizeof(error));
        if (!context) {
            status = TPGP_EBACKEND;
            goto buffers;
        }
        tpgp_cuda_status cs = tpgp_cuda_branches(
            context, map->keys, map->values, map->len, map->n_words, map->n_spins, i, j, qi, qj, c,
            s, rc, rs, o->validated, raw.keys, raw.values, rad, error, sizeof(error));
        if (!o->cuda_context)
            tpgp_cuda_destroy(context);
        if (cs != TPGP_CUDA_OK) {
            status = cs == TPGP_CUDA_ENOMEM     ? TPGP_ENOMEM
                     : cs == TPGP_CUDA_ENUMERIC ? TPGP_ENUMERIC
                                                : TPGP_EBACKEND;
            goto buffers;
        }
    } else if (o->backend == TPGP_PTHREAD && nt > 1) {
        pthread_t *threads = calloc(nt, sizeof(*threads));
        branch_job *jobs = calloc(nt, sizeof(*jobs));
        if (!threads || !jobs) {
            free(threads);
            free(jobs);
            status = TPGP_ENOMEM;
            goto buffers;
        }
        unsigned made = 0;
        for (unsigned k = 0; k < nt; k++) {
            jobs[k] = base;
            jobs[k].start = map->len * k / nt;
            jobs[k].end = map->len * (k + 1) / nt;
            if (pthread_create(threads + k, NULL, branches, jobs + k)) {
                status = TPGP_EBACKEND;
                break;
            }
            made++;
        }
        for (unsigned k = 0; k < made; k++)
            pthread_join(threads[k], NULL);
        free(threads);
        free(jobs);
        if (status)
            goto buffers;
    }
#ifdef _OPENMP
    else if (o->backend == TPGP_OPENMP && nt > 1) {
#pragma omp parallel num_threads(nt)
        {
            branch_job job = base;
            size_t tid = (size_t)omp_get_thread_num(), n = (size_t)omp_get_num_threads();
            job.start = map->len * tid / n;
            job.end = map->len * (tid + 1) / n;
            branches(&job);
        }
    }
#endif
    else
        branches(&base);
    for (size_t k = 0; k < count; k++) {
        if (!isfinite(raw.values[k]) || !isfinite(rad[k])) {
            status = TPGP_ENUMERIC;
            goto buffers;
        }
        idx[k] = k;
    }
    stable_sort_indices(idx, tmp, 0, count, &raw);
    status = reserve(&out, count);
    if (status)
        goto buffers;
    size_t discarded = 0, nrad = 0;
    double l1 = 0, sq = 0;
    for (size_t k = 0; k < count;) {
        size_t p = idx[k++];
        double v = raw.values[p], r = rad[p];
        const uint64_t *key = raw.keys + p * raw.n_words;
        while (k < count && keycmp(key, raw.keys + idx[k] * raw.n_words, raw.n_words) == 0) {
            size_t q = idx[k++];
            double previous = v;
            v += raw.values[q];
            r = upadd(r, rad[q]);
            if (o->validated && previous != 0 && raw.values[q] != 0)
                r = upadd(r, round_radius(v));
        }
        if (!isfinite(v) || !isfinite(r)) {
            status = TPGP_ENUMERIC;
            goto buffers;
        }
        if (r)
            merged_rad[nrad++] = r;
        int identity = 1;
        for (size_t w = 0; w < raw.n_words; w++)
            if (key[w]) {
                identity = 0;
                break;
            }
        if (v == 0)
            continue;
        if (!identity && fabs(v) <= o->threshold) {
            removed[discarded++] = fabs(v);
            l1 = upadd(l1, fabs(v));
            sq = upadd(sq, upmul(v, v));
            continue;
        }
        memcpy(out.keys + out.len * out.n_words, key, out.n_words * sizeof(uint64_t));
        out.values[out.len++] = v;
    }
    if (o->max_terms && out.len > o->max_terms) {
        status = TPGP_ECAP;
        goto buffers;
    }
    tpgp_ledger next = *ledger;
    double l2 = upper_l2(removed, discarded), ar = upper_l2(merged_rad, nrad);
    next.dropped_l1 = upadd(next.dropped_l1, l1);
    next.dropped_l2 = upadd(next.dropped_l2, l2);
    next.dropped_l2_squared = upadd(next.dropped_l2_squared, sq);
    next.arithmetic_l2 = upadd(next.arithmetic_l2, ar);
    next.rotations++;
    if (discarded)
        next.removal_events++;
    if (out.len > next.peak_terms)
        next.peak_terms = out.len;
    next.arithmetic_enclosed = next.arithmetic_enclosed && o->validated && tpgp_has_mpfr();
    if (!isfinite(next.dropped_l2) || !isfinite(next.arithmetic_l2)) {
        status = TPGP_ENUMERIC;
        goto buffers;
    }
    if (o->max_drop_l2 && next.dropped_l2 > o->max_drop_l2) {
        status = TPGP_EBUDGET;
        goto buffers;
    }
    tpgp_map_free(map);
    *map = out;
    memset(&out, 0, sizeof(out));
    *ledger = next;
buffers:
    free(rad);
    free(idx);
    free(tmp);
    free(removed);
    free(merged_rad);
    free(mk);
    free(ms);
    free(ma);
done:
    tpgp_map_free(&raw);
    tpgp_map_free(&out);
    return status;
}
tpgp_status tpgp_rotate(tpgp_map *m, size_t i, size_t j, unsigned qi, unsigned qj, double theta,
                        const tpgp_options *o, tpgp_ledger *l) {
    return rotate_product(m, i, j, qi, qj, theta, 1.0, o, l);
}
tpgp_status tpgp_strang_step(tpgp_map *m, const tpgp_term *terms, size_t n, double dt,
                             const tpgp_options *o, tpgp_ledger *l) {
    if (!m || (!terms && n) || !o || !l || !isfinite(dt))
        return TPGP_EINVAL;
    /* Multiplication by 1/2 is exact unless dt is subnormal and odd. Reject
       that case rather than silently changing the declared Hamiltonian time. */
    double half = dt * 0.5;
    if (half * 2 != dt)
        return TPGP_ENUMERIC;
    tpgp_map tmp = {0};
    tpgp_status s = tpgp_map_clone(&tmp, m);
    if (s)
        return s;
    tpgp_ledger ledger = *l;
    for (size_t k = 0; k < n; k++) {
        const tpgp_term *t = terms + k;
        s = rotate_product(&tmp, t->i, t->j, t->qi, t->qj, t->h, half, o, &ledger);
        if (s)
            goto fail;
    }
    for (size_t k = n; k > 0; k--) {
        const tpgp_term *t = terms + k - 1;
        s = rotate_product(&tmp, t->i, t->j, t->qi, t->qj, t->h, half, o, &ledger);
        if (s)
            goto fail;
    }
    tpgp_map_free(m);
    *m = tmp;
    *l = ledger;
    return TPGP_OK;
fail:
    tpgp_map_free(&tmp);
    return s;
}
tpgp_status tpgp_liouvillian(tpgp_map *out, const tpgp_map *in, const tpgp_term *terms, size_t n) {
    if (!out || !in || (!terms && n))
        return TPGP_EINVAL;
    tpgp_map tmp = {0};
    tpgp_status s = tpgp_map_init(&tmp, in->n_spins);
    if (s)
        return s;
    uint64_t *key = malloc(in->n_words * sizeof(uint64_t));
    if (!key) {
        tpgp_map_free(&tmp);
        return TPGP_ENOMEM;
    }
    for (size_t t = 0; t < n; t++) {
        const tpgp_term *q = terms + t;
        if (q->i >= in->n_spins || q->j >= in->n_spins || q->i == q->j || q->qi > 3 || q->qj > 3 ||
            !isfinite(q->h)) {
            s = TPGP_EINVAL;
            goto done;
        }
        for (size_t k = 0; k < in->len; k++) {
            const uint64_t *p = in->keys + k * in->n_words;
            unsigned pi = tpgp_key_get(p, q->i), pj = tpgp_key_get(p, q->j);
            if (!(anticomm(q->qi, pi) ^ anticomm(q->qj, pj)))
                continue;
            unsigned phase = (tpgp_pauli_phase(q->qi, pi) + tpgp_pauli_phase(q->qj, pj)) & 3u;
            double v = 2 * q->h * in->values[k] * (phase == 1 ? 1 : -1);
            if (!isfinite(v)) {
                s = TPGP_ENUMERIC;
                goto done;
            }
            memcpy(key, p, in->n_words * sizeof(uint64_t));
            tpgp_key_set(key, q->i, pi ^ q->qi);
            tpgp_key_set(key, q->j, pj ^ q->qj);
            s = tpgp_map_add(&tmp, key, v);
            if (s)
                goto done;
        }
    }
done:
    free(key);
    if (s) {
        tpgp_map_free(&tmp);
        return s;
    }
    tpgp_map_free(out);
    *out = tmp;
    return TPGP_OK;
}
static int term_anticommutes(const tpgp_term *a, const tpgp_term *b) {
    int parity = 0;
    if (a->i == b->i)
        parity ^= anticomm(a->qi, b->qi);
    if (a->i == b->j)
        parity ^= anticomm(a->qi, b->qj);
    if (a->j == b->i)
        parity ^= anticomm(a->qj, b->qi);
    if (a->j == b->j)
        parity ^= anticomm(a->qj, b->qj);
    return parity;
}
typedef struct {
    size_t *v, n, cap;
} neighbor_list;
static int neighbor_add(neighbor_list *list, size_t x) {
    if (list->n == list->cap) {
        size_t cap = list->cap ? 2 * list->cap : 8;
        if (cap < list->cap || cap > SIZE_MAX / sizeof(size_t))
            return 0;
        size_t *p = realloc(list->v, cap * sizeof(size_t));
        if (!p)
            return 0;
        list->v = p;
        list->cap = cap;
    }
    list->v[list->n++] = x;
    return 1;
}
double tpgp_strang_bound_taylor(const tpgp_term *terms, size_t n, double dt) {
    /* Hamiltonian unitary Taylor remainders each cost lambda_H^3 |dt|^3/6.
       Conjugation is 2-Lipschitz in the unitary operator norm, giving 2/3
       overall. Applying Taylor directly in Liouville space is 4x looser. */
    if ((!terms && n) || !isfinite(dt))
        return NAN;
    double lambda = 0;
    for (size_t k = 0; k < n; k++) {
        if (!isfinite(terms[k].h) || terms[k].i == terms[k].j || terms[k].qi > 3 || terms[k].qj > 3)
            return NAN;
        lambda = upadd(lambda, fabs(terms[k].h));
    }
    double x = upmul(lambda, fabs(dt));
    return upmul(upmul(upmul(x, x), x), nextafter(2.0 / 3.0, INFINITY));
}
double tpgp_strang_bound(const tpgp_term *terms, size_t n, double dt) {
    double general = tpgp_strang_bound_taylor(terms, n, dt);
    if (isnan(general))
        return NAN;
    neighbor_list *adj = calloc(n, sizeof(*adj));
    if (!adj)
        return general;
    int ok = 1;
    for (size_t a = 0; a < n && ok; a++)
        for (size_t b = a + 1; b < n; b++)
            if (terms[a].h && terms[b].h && term_anticommutes(terms + a, terms + b)) {
                if (!neighbor_add(adj + a, b) || !neighbor_add(adj + b, a)) {
                    ok = 0;
                    break;
                }
            }
    double C = 0;
    for (size_t a = 0; a < n && ok; a++) {
        double ha = fabs(terms[a].h), inner = 0;
        for (size_t k = 0; k < adj[a].n; k++)
            if (adj[a].v[k] > a)
                inner = upadd(inner, fabs(terms[adj[a].v[k]].h));
        C = upadd(C, upmul(upmul(upmul(ha, ha), inner), nextafter(1.0 / 6.0, INFINITY)));
        for (size_t k = 0; k < adj[a].n; k++) {
            size_t b = adj[a].v[k];
            if (b <= a)
                continue;
            size_t ia = 0, ib = 0;
            double sum = 0;
            /* c anticommutes with Pa Pb iff it anticommutes with exactly
               one factor. Sorted-neighbor symmetric difference avoids a
               dense cubic loop on finite-degree interaction graphs. */
            while (ia < adj[a].n || ib < adj[b].n) {
                size_t c;
                if (ib == adj[b].n || (ia < adj[a].n && adj[a].v[ia] < adj[b].v[ib]))
                    c = adj[a].v[ia++];
                else if (ia == adj[a].n || adj[b].v[ib] < adj[a].v[ia])
                    c = adj[b].v[ib++];
                else {
                    ia++;
                    ib++;
                    continue;
                }
                if (c > a)
                    sum = upadd(sum, fabs(terms[c].h));
            }
            C = upadd(
                C, upmul(upmul(upmul(ha, fabs(terms[b].h)), sum), nextafter(1.0 / 3.0, INFINITY)));
        }
    }
    for (size_t k = 0; k < n; k++)
        free(adj[k].v);
    free(adj);
    if (!ok)
        return general;
    /* Hamiltonian nested commutators have constants 1/24 and 1/12;
       Pauli anticommutators supply factor four, and conjugation adds two. */
    double t = fabs(dt), nested = upmul(upmul(upmul(upmul(2, C), t), t), t);
    return fmin(general, nested);
}
