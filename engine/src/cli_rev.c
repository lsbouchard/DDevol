#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "checkpoint_rev.h"
#include "residual_rev.h"
#include "tpgp_cuda_rev.h"
#include "tpgp_metal_rev.h"
#include "tpgp_rev.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    tpgp_term *data;
    size_t len, cap;
} term_list;
static volatile sig_atomic_t stop_requested = 0;
static void stop_handler(int signal_number) {
    (void)signal_number;
    stop_requested = 1;
}
static void release_backends(tpgp_options *o) {
    tpgp_metal_destroy(o->metal_context);
    tpgp_cuda_destroy(o->cuda_context);
    o->metal_context = NULL;
    o->cuda_context = NULL;
}
static int lock_checkpoint(const char *checkpoint_path, const char *restart_path) {
    size_t length = strlen(checkpoint_path);
    if (length > SIZE_MAX - 6)
        return -1;
    char *lock_path = malloc(length + 6);
    if (!lock_path)
        return -1;
    snprintf(lock_path, length + 6, "%s.lock", checkpoint_path);
    int fd = open(lock_path, O_CREAT | O_RDWR, 0600);
    free(lock_path);
    if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB)) {
        fprintf(stderr, "Checkpoint is locked by another run or cannot be locked.\n");
        if (fd >= 0)
            close(fd);
        return -1;
    }
    struct stat existing;
    if (lstat(checkpoint_path, &existing) == 0 &&
        (!restart_path || strcmp(checkpoint_path, restart_path))) {
        fprintf(stderr, "Refusing to overwrite an existing checkpoint without its restart.\n");
        close(fd);
        return -1;
    }
    return fd;
}
static unsigned option_flag(const char *a) {
    static const char *names[] = {
        "--model",     "--initial",        "--n",       "--linear-size", "--cap",
        "--threads",   "--residual-depth", "--dt",      "--time",        "--j2",
        "--threshold", "--drop-budget",    "--backend", "--validated",   "--edges",
        "--max-fid-bound"};
    for (unsigned k = 0; k < sizeof(names) / sizeof(names[0]); k++)
        if (!strcmp(a, names[k]))
            return 1u << k;
    return 0;
}
static tpgp_status save_current(const char *path, const tpgp_map *map, const term_list *terms,
                                const tpgp_options *options, const tpgp_ledger *ledger, double dt,
                                double pf, double residual, size_t step, size_t steps, int depth,
                                int global, const char *model, double j2, size_t L,
                                size_t resumed_from, size_t prior_steps, size_t prior_cap,
                                int reason, tpgp_status last_status, double max_fid_bound) {
    tpgp_checkpoint s = {0};
    s.map = *map;
    s.terms = terms->data;
    s.term_count = terms->len;
    s.options = *options;
    s.ledger = *ledger;
    s.dt = dt;
    s.pf_bound = pf;
    s.residual_bound = residual;
    s.step = step;
    s.total_steps = steps;
    s.residual_depth = depth;
    s.initial_global = global;
    snprintf(s.model, sizeof(s.model), "%s", model);
    s.j2 = j2;
    s.linear_size = L;
    s.resumed_from_step = resumed_from;
    s.prior_total_steps = prior_steps;
    s.prior_max_terms = prior_cap;
    s.stop_reason = reason;
    s.last_status = last_status;
    s.max_fid_bound = max_fid_bound;
    char error[512];
    tpgp_status status = tpgp_checkpoint_save(path, &s, error, sizeof(error));
    fprintf(stderr, "checkpoint step=%zu/%zu path=%s status=%s stop_reason=%d\n", step, steps, path,
            error, reason);
    return status;
}
static int append(term_list *l, size_t i, size_t j, unsigned a, unsigned b, double h) {
    if (h == 0)
        return 1;
    if (l->len == l->cap) {
        size_t cap = l->cap ? 2 * l->cap : 64;
        if (cap < l->cap || cap > SIZE_MAX / sizeof(tpgp_term))
            return 0;
        tpgp_term *p = realloc(l->data, cap * sizeof(*p));
        if (!p)
            return 0;
        l->data = p;
        l->cap = cap;
    }
    tpgp_term t = {i, j, a, b, h, l->len};
    l->data[l->len++] = t;
    return 1;
}
static int edge(term_list *l, size_t i, size_t j, double d, const char *model) {
    if (!strcmp(model, "ising"))
        return append(l, i, j, 3, 3, d / 4);
    if (!strcmp(model, "xy"))
        return append(l, i, j, 1, 1, d / 4) && append(l, i, j, 2, 2, d / 4);
    return append(l, i, j, 3, 3, d / 2) && append(l, i, j, 1, 1, -d / 4) &&
           append(l, i, j, 2, 2, -d / 4);
}
static int number(const char *s, double *v) {
    char *end;
    errno = 0;
    *v = strtod(s, &end);
    return !errno && end != s && !*end && isfinite(*v);
}
static int integer(const char *s, size_t *v) {
    char *end;
    errno = 0;
    if (*s == '-')
        return 0;
    uintmax_t x = strtoumax(s, &end, 10);
    if (errno || end == s || *end || x > SIZE_MAX)
        return 0;
    *v = (size_t)x;
    return 1;
}
static double wall(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static double rss_mb(void) {
    struct rusage r;
    if (getrusage(RUSAGE_SELF, &r))
        return NAN;
#ifdef __APPLE__
    return r.ru_maxrss / (1024.0 * 1024.0);
#else
    return r.ru_maxrss / 1024.0;
#endif
}
static void usage(FILE *f) {
    fprintf(f, "Usage: tpgp_rev [--model ring|j1j2|ising|xy|cubic|edges] [--n N]\n"
               "  --dt STEP --time END --threshold X --cap TERMS --drop-budget X\n"
               "  --backend serial|pthread|openmp|metal|cuda --threads N --validated\n"
               "  --j2 RATIO --linear-size L --edges PATH --initial global|local\n"
               "  --residual-depth 0..10 (independent MPFR endpoint bound; global initial only)\n"
               "  --output FILE (default stdout)\n"
               "  --checkpoint FILE --checkpoint-every STEPS --checkpoint-seconds SECONDS\n"
               "  --restart FILE --walltime SECONDS (stops only at accepted step boundaries)\n"
               "  --max-fid-bound X (stop if the best complete FID bound exceeds X)\n"
               "  A checkpointed pause exits 75; a completed trajectory exits 0.\n"
               "  An accuracy-limit stop exits 76 and requires a checkpoint path.\n"
               "Hamiltonian: secular d(2ZZ-XX-YY)/4; Ising d ZZ/4; XY d(XX+YY)/4.\n"
               "Cubic is periodic [001], nearest and face-diagonal shells, lattice units.\n"
               "Edge file rows: i j d (zero-based; secular coupling). No hidden fallback.\n");
}
static int models(term_list *terms, const char *model, size_t n, size_t L, double j2,
                  const char *path) {
    if (!strcmp(model, "cubic")) {
        if (L < 3 || L > SIZE_MAX / L || L * L > SIZE_MAX / L || L * L * L != n)
            return 0;
        for (size_t i = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++) {
                long a[3] = {(long)(i % L), (long)((i / L) % L), (long)(i / (L * L))},
                     b[3] = {(long)(j % L), (long)((j / L) % L), (long)(j / (L * L))};
                long r2 = 0, dz = 0;
                for (int k = 0; k < 3; k++) {
                    long d = b[k] - a[k];
                    if (d > (long)L / 2)
                        d -= (long)L;
                    if (d < -(long)L / 2)
                        d += (long)L;
                    r2 += d * d;
                    if (k == 2)
                        dz = d;
                }
                if (r2 == 1 || r2 == 2) {
                    double coupling = (1 - 3.0 * dz * dz / r2) / (r2 * sqrt((double)r2));
                    if (!edge(terms, i, j, coupling, "ring"))
                        return 0;
                }
            }
        return 1;
    }
    if (!strcmp(model, "edges")) {
        if (!path)
            return 0;
        FILE *f = fopen(path, "r");
        if (!f)
            return 0;
        char line[512];
        int ok = 1;
        while (fgets(line, sizeof(line), f)) {
            size_t i, j;
            double d;
            char extra;
            if (line[0] == '#' || line[0] == '\n')
                continue;
            if (sscanf(line, "%zu %zu %lf %c", &i, &j, &d, &extra) != 3 || i >= n || j >= n ||
                i == j || !isfinite(d) || !edge(terms, i, j, d, "ring")) {
                ok = 0;
                break;
            }
        }
        if (ferror(f))
            ok = 0;
        fclose(f);
        return ok;
    }
    if (strcmp(model, "ring") && strcmp(model, "j1j2") && strcmp(model, "ising") &&
        strcmp(model, "xy"))
        return 0;
    if (n < 3 || (!strcmp(model, "j1j2") && n < 5))
        return 0;
    for (size_t i = 0; i < n; i++) {
        if (!edge(terms, i, (i + 1) % n, 1, model))
            return 0;
        if (!strcmp(model, "j1j2") && !edge(terms, i, (i + 2) % n, j2, model))
            return 0;
    }
    return 1;
}
int main(int argc, char **argv) {
    const char *model = "ring", *initial = "global", *output = NULL, *edge_path = NULL;
    const char *checkpoint_path = NULL, *restart_path = NULL;
    size_t checkpoint_every = 100;
    double checkpoint_seconds = 0, walltime = 0, max_fid_bound = 0;
    unsigned explicit_options = 0;
    size_t n = 6, L = 3;
    double dt = .01, T = .1, j2 = .3;
    int residual_depth = -1;
    tpgp_options o = tpgp_default_options();
    o.max_terms = 1000000;
    for (int k = 1; k < argc; k++) {
        const char *a = argv[k];
        explicit_options |= option_flag(a);
        if (!strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        }
        if (!strcmp(a, "--validated")) {
            o.validated = 1;
            continue;
        }
        if (k + 1 == argc) {
            usage(stderr);
            return 2;
        }
        const char *v = argv[++k];
        size_t z;
        if (!strcmp(a, "--model"))
            model = v;
        else if (!strcmp(a, "--initial"))
            initial = v;
        else if (!strcmp(a, "--output"))
            output = v;
        else if (!strcmp(a, "--edges"))
            edge_path = v;
        else if (!strcmp(a, "--checkpoint"))
            checkpoint_path = v;
        else if (!strcmp(a, "--restart"))
            restart_path = v;
        else if (!strcmp(a, "--checkpoint-every")) {
            if (!integer(v, &checkpoint_every) || !checkpoint_every)
                return 2;
        } else if (!strcmp(a, "--checkpoint-seconds")) {
            if (!number(v, &checkpoint_seconds) || checkpoint_seconds < 0)
                return 2;
        } else if (!strcmp(a, "--walltime")) {
            if (!number(v, &walltime) || walltime < 0)
                return 2;
        } else if (!strcmp(a, "--max-fid-bound")) {
            if (!number(v, &max_fid_bound) || max_fid_bound <= 0)
                return 2;
        } else if (!strcmp(a, "--n")) {
            if (!integer(v, &n))
                return 2;
        } else if (!strcmp(a, "--linear-size")) {
            if (!integer(v, &L))
                return 2;
        } else if (!strcmp(a, "--cap")) {
            if (!integer(v, &o.max_terms))
                return 2;
        } else if (!strcmp(a, "--threads")) {
            if (!integer(v, &z) || !z || z > 65536)
                return 2;
            o.threads = (unsigned)z;
        } else if (!strcmp(a, "--residual-depth")) {
            if (!integer(v, &z) || z > 10)
                return 2;
            residual_depth = (int)z;
        } else if (!strcmp(a, "--dt")) {
            if (!number(v, &dt))
                return 2;
        } else if (!strcmp(a, "--time")) {
            if (!number(v, &T))
                return 2;
        } else if (!strcmp(a, "--j2")) {
            if (!number(v, &j2))
                return 2;
        } else if (!strcmp(a, "--threshold")) {
            if (!number(v, &o.threshold))
                return 2;
        } else if (!strcmp(a, "--drop-budget")) {
            if (!number(v, &o.max_drop_l2))
                return 2;
        } else if (!strcmp(a, "--backend")) {
            if (!strcmp(v, "serial"))
                o.backend = TPGP_SERIAL;
            else if (!strcmp(v, "pthread"))
                o.backend = TPGP_PTHREAD;
            else if (!strcmp(v, "openmp"))
                o.backend = TPGP_OPENMP;
            else if (!strcmp(v, "metal"))
                o.backend = TPGP_METAL;
            else if (!strcmp(v, "cuda"))
                o.backend = TPGP_CUDA;
            else
                return 2;
        } else {
            fprintf(stderr, "Unknown option: %s\n", a);
            return 2;
        }
    }
    tpgp_checkpoint loaded = {0};
    size_t prior_steps = 0, prior_cap = 0;
    if (restart_path && !checkpoint_path)
        checkpoint_path = restart_path;
    /* Acquire before reading a restart record: concurrent runs must not load
       an old generation and later overwrite a newer accepted checkpoint. */
    int checkpoint_lock = -1;
    if (checkpoint_path) {
        checkpoint_lock = lock_checkpoint(checkpoint_path, restart_path);
        if (checkpoint_lock < 0)
            return 2;
    }
    if (restart_path) {
        char error[512];
        tpgp_status loaded_status =
            tpgp_checkpoint_load(restart_path, &loaded, error, sizeof(error));
        if (loaded_status) {
            fprintf(stderr, "Restart rejected: %s\n", error);
            return 2;
        }
        int mismatch = 0;
        if ((explicit_options & (1u << 0)) && strcmp(model, loaded.model))
            mismatch = 1;
        if ((explicit_options & (1u << 1)) &&
            strcmp(initial, loaded.initial_global ? "global" : "local"))
            mismatch = 1;
        if ((explicit_options & (1u << 2)) && n != loaded.map.n_spins)
            mismatch = 1;
        if ((explicit_options & (1u << 3)) && L != loaded.linear_size)
            mismatch = 1;
        if ((explicit_options & (1u << 4)) && o.max_terms != 0 &&
            (loaded.options.max_terms == 0 || o.max_terms < loaded.options.max_terms))
            mismatch = 1;
        if ((explicit_options & (1u << 5)) && o.threads != loaded.options.threads)
            mismatch = 1;
        if ((explicit_options & (1u << 6)) && residual_depth != loaded.residual_depth)
            mismatch = 1;
        if ((explicit_options & (1u << 7)) && dt != loaded.dt)
            mismatch = 1;
        if ((explicit_options & (1u << 8)) && T < (double)loaded.total_steps * loaded.dt)
            mismatch = 1;
        if ((explicit_options & (1u << 9)) && j2 != loaded.j2)
            mismatch = 1;
        if ((explicit_options & (1u << 10)) && o.threshold != loaded.options.threshold)
            mismatch = 1;
        if ((explicit_options & (1u << 11)) && o.max_drop_l2 != loaded.options.max_drop_l2)
            mismatch = 1;
        if ((explicit_options & (1u << 12)) && o.backend != loaded.options.backend)
            mismatch = 1;
        if ((explicit_options & (1u << 13)) && o.validated != loaded.options.validated)
            mismatch = 1;
        if (mismatch) {
            fprintf(stderr, "Restart rejected: explicitly supplied model/numerical options differ "
                            "from the checkpoint.\n");
            tpgp_checkpoint_free(&loaded);
            return 2;
        }
        size_t requested_cap = o.max_terms;
        double requested_time = T;
        prior_steps = loaded.total_steps;
        prior_cap = loaded.options.max_terms;
        model = loaded.model;
        initial = loaded.initial_global ? "global" : "local";
        n = loaded.map.n_spins;
        L = loaded.linear_size;
        dt = loaded.dt;
        T = (explicit_options & (1u << 8)) ? requested_time : (double)loaded.total_steps * dt;
        j2 = loaded.j2;
        o = loaded.options;
        residual_depth = loaded.residual_depth;
        if (explicit_options & (1u << 4))
            o.max_terms = requested_cap;
        if (!(explicit_options & (1u << 15)))
            max_fid_bound = loaded.max_fid_bound;
        if (!checkpoint_path)
            checkpoint_path = restart_path;
    }
    if (((walltime > 0 || max_fid_bound > 0) && !checkpoint_path) ||
        (checkpoint_path && output && !strcmp(checkpoint_path, output))) {
        fprintf(stderr, "Walltime/accuracy stops require a checkpoint path, distinct from CSV.\n");
        tpgp_checkpoint_free(&loaded);
        return 2;
    }
    if (!strcmp(model, "cubic")) {
        if (L < 3 || L > 10000 || L > SIZE_MAX / L || L * L > SIZE_MAX / L)
            return 2;
        n = L * L * L;
    }
    if (!n || dt <= 0 || T < 0 || o.threshold < 0 || o.max_drop_l2 < 0 ||
        (strcmp(initial, "local") && strcmp(initial, "global"))) {
        usage(stderr);
        return 2;
    }
    double ratio = T / dt, rounded = round(ratio);
    if (!isfinite(ratio) || rounded >= (double)SIZE_MAX ||
        fabs(ratio - rounded) > 32 * DBL_EPSILON * fmax(1, fabs(ratio))) {
        fprintf(stderr,
                "END must be an integer multiple of STEP; no endpoint is silently shortened.\n");
        return 2;
    }
    size_t steps =
        restart_path && !(explicit_options & (1u << 8)) ? loaded.total_steps : (size_t)rounded;
    if (o.validated && !tpgp_has_mpfr()) {
        fprintf(stderr, "Validated mode requires MPFR at build time.\n");
        return 2;
    }
    if (residual_depth >= 0 && (!tpgp_has_mpfr() || strcmp(initial, "global"))) {
        fprintf(stderr, "Residual CLI bounds require MPFR and global initial sum Xi.\n");
        return 2;
    }
    if (max_fid_bound > 0 &&
        (!tpgp_has_mpfr() || (!o.validated && residual_depth < 0))) {
        fprintf(stderr, "An accuracy stop requires a complete MPFR-enclosed error bound.\n");
        tpgp_checkpoint_free(&loaded);
        return 2;
    }
    if (o.backend == TPGP_OPENMP && !tpgp_has_openmp()) {
        fprintf(stderr, "OpenMP backend not built.\n");
        return 2;
    }
    if (o.backend == TPGP_METAL) {
        char err[512];
        o.metal_context = tpgp_metal_create(err, sizeof(err));
        if (!o.metal_context) {
            fprintf(stderr, "Metal unavailable: %s\n", err);
            return 2;
        }
    }
    if (o.backend == TPGP_CUDA) {
        char error[512];
        o.cuda_context = tpgp_cuda_create(0, error, sizeof(error));
        if (!o.cuda_context) {
            fprintf(stderr, "CUDA unavailable: %s\n", error);
            tpgp_checkpoint_free(&loaded);
            return 2;
        }
    }
    term_list terms = {0};
    if (restart_path && !edge_path) {
        terms.data = loaded.terms;
        terms.len = terms.cap = loaded.term_count;
        loaded.terms = NULL;
        loaded.term_count = 0;
    } else if (!models(&terms, model, n, L, j2, edge_path)) {
        fprintf(stderr, "Invalid model geometry/coupling file or allocation failure.\n");
        free(terms.data);
        release_backends(&o);
        tpgp_checkpoint_free(&loaded);
        return 2;
    }
    if (restart_path && edge_path) {
        tpgp_checkpoint requested = loaded;
        requested.terms = terms.data;
        requested.term_count = terms.len;
        if (!tpgp_checkpoint_same_configuration(&loaded, &requested)) {
            fprintf(stderr, "Restart rejected: coupling file differs from saved Hamiltonian.\n");
            free(terms.data);
            release_backends(&o);
            tpgp_checkpoint_free(&loaded);
            return 2;
        }
        free(loaded.terms);
        loaded.terms = NULL;
        loaded.term_count = 0;
    }
    tpgp_map m = {0};
    tpgp_status status = TPGP_OK;
    if (restart_path) {
        m = loaded.map;
        memset(&loaded.map, 0, sizeof(loaded.map));
    } else
        status = tpgp_map_init(&m, n);
    size_t input_terms = !strcmp(initial, "global") ? n : 1;
    for (size_t i = 0; !restart_path && i < input_terms && !status; i++)
        status = tpgp_map_add_single(&m, i, 1, 1);
    if (status || (o.max_terms && m.len > o.max_terms)) {
        fprintf(stderr, "Initial state failed: %s\n",
                status ? tpgp_status_string(status) : "active-term cap exceeded");
        free(terms.data);
        tpgp_map_free(&m);
        release_backends(&o);
        tpgp_checkpoint_free(&loaded);
        return 2;
    }
    FILE *f = output ? fopen(output, "wx") : stdout;
    if (!f) {
        perror(output);
        free(terms.data);
        tpgp_map_free(&m);
        release_backends(&o);
        tpgp_checkpoint_free(&loaded);
        return 2;
    }
    tpgp_ledger ledger;
    tpgp_ledger_init(&ledger, o.validated);
    ledger.peak_terms = m.len;
    if (restart_path)
        ledger = loaded.ledger;
    size_t first_step = restart_path ? loaded.step : 0, accepted_step = first_step,
           last_saved = SIZE_MAX;
    double start = wall(), last_checkpoint = start, pf = restart_path ? loaded.pf_bound : 0,
           per_step = tpgp_strang_bound(terms.data, terms.len, dt),
           norm0 = sqrt((double)input_terms), residual_total = residual_depth >= 0 ? 0 : NAN;
    if (restart_path)
        residual_total = loaded.residual_bound;
    int paused = 0, accuracy_stopped = 0;
    signal(SIGINT, stop_handler);
    signal(SIGTERM, stop_handler);
    fprintf(f,
            "# "
            "model=%s,n=%zu,initial=%s,hamiltonian_terms=%zu,dt=%.17g,threshold=%.17g,backend=%d,"
            "threads=%u,validated=%d\n",
            model, n, initial, terms.len, dt, o.threshold, o.backend, o.threads, o.validated);
    fprintf(f, "# Initial coefficients are exactly one; normalization is only at readout. Bounds "
               "refer to binary64 Hamiltonian inputs.\n");
    fprintf(f, "# restart_step=%zu,prior_total_steps=%zu,total_steps=%zu,prior_cap=%zu,cap=%zu,"
               "max_fid_bound=%.17g,prior_max_fid_bound=%.17g\n",
            first_step, prior_steps, steps, prior_cap, o.max_terms, max_fid_bound,
            restart_path ? loaded.max_fid_bound : 0);
    fprintf(f,
            "# Row k represents exact physical time k times the binary64 dt input; the time "
            "column is its rounded display value. dt_hex=%a\n",
            dt);
    fprintf(f, "time,fid,active,l2_norm,drop_l1,drop_l2_sum,drop_l2_squared,arithmetic_l2,pf_"
               "observable_bound,total_observable_bound,residual_raw_bound,residual_fid_bound,"
               "readout_radius,arithmetic_enclosed,wall_seconds,peak_rss_mb\n");
    for (size_t step = first_step; step <= steps; step++) {
        double fid = tpgp_map_single_sum(&m, 1) / (double)input_terms, readout_radius = NAN;
        if (tpgp_has_mpfr()) {
            status = tpgp_readout(&m, 1, input_terms, &fid, &readout_radius);
            if (status)
                break;
        }
        /* Global FID dual norm is sqrt(N)/N. Local-source summed-X response
           has dual norm sqrt(N), so its bound uses a different factor. */
        double dual = sqrt((double)n) / (double)input_terms;
        double pf_obs = pf * norm0 * dual;
        double bound = pf_obs + (ledger.dropped_l2 + ledger.arithmetic_l2) * dual;
        if (tpgp_has_mpfr()) {
            double loss = tpgp_bound_add_up(ledger.dropped_l2, ledger.arithmetic_l2);
            if (input_terms == n) {
                pf_obs = pf;
                bound = tpgp_bound_add_up(pf, tpgp_bound_fid_up(loss, n));
            } else {
                pf_obs = pf == 0 ? 0 : nextafter(tpgp_bound_fid_up(pf, n) * (double)n, INFINITY);
                bound = nextafter(tpgp_bound_fid_up(tpgp_bound_add_up(pf, loss), n) * (double)n,
                                  INFINITY);
            }
            bound = tpgp_bound_add_up(bound, readout_radius);
        }
        double residual_fid =
            residual_depth >= 0
                ? tpgp_bound_add_up(tpgp_bound_fid_up(residual_total, n), readout_radius)
                : NAN;
        fprintf(f,
                "%.17g,%.17g,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%."
                "6f,%.3f\n",
                step * dt, fid, m.len, tpgp_map_l2(&m), ledger.dropped_l1, ledger.dropped_l2,
                ledger.dropped_l2_squared, ledger.arithmetic_l2, pf_obs, bound, residual_total,
                residual_fid, readout_radius, ledger.arithmetic_enclosed, wall() - start, rss_mb());
        if (fflush(f)) {
            status = TPGP_EINVAL;
            break;
        }
        double best_bound = o.validated && ledger.arithmetic_enclosed ? bound : INFINITY;
        if (residual_depth >= 0 && residual_fid < best_bound)
            best_bound = residual_fid;
        int reason = max_fid_bound > 0 && best_bound > max_fid_bound ? 6
                     : step == steps                                ? 1
                     : stop_requested                               ? 3
                     : (walltime > 0 && wall() - start >= walltime) ? 2
                                                                    : 0;
        int checkpoint_due =
            reason || step % checkpoint_every == 0 ||
            (checkpoint_seconds > 0 && wall() - last_checkpoint >= checkpoint_seconds);
        if (checkpoint_path && checkpoint_due && last_saved != step) {
            status = save_current(checkpoint_path, &m, &terms, &o, &ledger, dt, pf, residual_total,
                                  step, steps, residual_depth, input_terms == n, model, j2, L,
                                  first_step, prior_steps, prior_cap, reason, TPGP_OK, max_fid_bound);
            if (status)
                break;
            last_saved = step;
            last_checkpoint = wall();
        }
        if (reason) {
            paused = reason == 2 || reason == 3;
            accuracy_stopped = reason == 6;
            break;
        }
        tpgp_map previous = {0};
        tpgp_ledger previous_ledger = ledger;
        if (residual_depth >= 0) {
            status = tpgp_map_clone(&previous, &m);
            if (status)
                break;
        }
        status = tpgp_strang_step(&m, terms.data, terms.len, dt, &o, &ledger);
        if (status) {
            fprintf(stderr, "Step %zu rejected: %s; last accepted time %.17g, active=%zu.\n",
                    step + 1, tpgp_status_string(status), step * dt, m.len);
            tpgp_map_free(&previous);
            break;
        }
        if (residual_depth >= 0) {
            double increment = 0;
            status = tpgp_residual_bound(&previous, &m, terms.data, terms.len, dt,
                                         (unsigned)residual_depth, &increment);
            if (status) {
                fprintf(stderr, "Independent endpoint bound failed after step %zu: %s.\n", step + 1,
                        tpgp_status_string(status));
                tpgp_map_free(&m);
                m = previous;
                memset(&previous, 0, sizeof(previous));
                ledger = previous_ledger;
                break;
            }
            tpgp_map_free(&previous);
            residual_total = tpgp_bound_add_up(residual_total, increment);
        }
        pf = tpgp_bound_add_up(pf, per_step);
        accepted_step = step + 1;
    }
    if (status && checkpoint_path) {
        int reason = status == TPGP_ECAP || status == TPGP_ENOMEM || status == TPGP_EBUDGET ? 4 : 5;
        (void)save_current(checkpoint_path, &m, &terms, &o, &ledger, dt, pf, residual_total,
                           accepted_step, steps, residual_depth, input_terms == n, model, j2, L,
                           first_step, prior_steps, prior_cap, reason, status, max_fid_bound);
    }
    fprintf(stderr,
            "status=%s accepted_rotations=%zu peak_terms=%zu elapsed=%.3fs peak_rss=%.3fMiB "
            "arithmetic_enclosed=%d completion=%s step=%zu/%zu\n",
            tpgp_status_string(status), ledger.rotations, ledger.peak_terms, wall() - start,
            rss_mb(), ledger.arithmetic_enclosed,
            status   ? "failed"
            : accuracy_stopped ? "accuracy_limit"
            : paused ? "paused"
                     : "complete",
            accepted_step, steps);
    if (ferror(f))
        status = TPGP_EINVAL;
    if (output && fclose(f))
        status = TPGP_EINVAL;
    free(terms.data);
    tpgp_map_free(&m);
    release_backends(&o);
    tpgp_checkpoint_free(&loaded);
    if (checkpoint_lock >= 0)
        close(checkpoint_lock);
    return status ? 1 : accuracy_stopped ? 76 : paused ? 75 : 0;
}
