#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "checkpoint_rev.h"
#include "residual_rev.h"
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

static int checks, failures;
#define CHECK(x) do { checks++; if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); failures++; } } while (0)
static int equal_map(const tpgp_map *a, const tpgp_map *b) {
    return a->n_spins == b->n_spins && a->n_words == b->n_words && a->len == b->len &&
           (!a->len || (!memcmp(a->keys, b->keys, a->len * a->n_words * sizeof(uint64_t)) &&
                        !memcmp(a->values, b->values, a->len * sizeof(double))));
}
static int equal_double(double a, double b) { return !memcmp(&a, &b, sizeof(double)); }
static int equal_ledger(const tpgp_ledger *a, const tpgp_ledger *b) {
    return equal_double(a->dropped_l1, b->dropped_l1) &&
           equal_double(a->dropped_l2, b->dropped_l2) &&
           equal_double(a->dropped_l2_squared, b->dropped_l2_squared) &&
           equal_double(a->arithmetic_l2, b->arithmetic_l2) &&
           a->rotations == b->rotations && a->removal_events == b->removal_events &&
           a->peak_terms == b->peak_terms && a->arithmetic_enclosed == b->arithmetic_enclosed;
}
static void initialize(tpgp_checkpoint *s, size_t n, double threshold) {
    CHECK(tpgp_map_init(&s->map, n) == TPGP_OK);
    for (size_t k = 0; k < n; k++)
        CHECK(tpgp_map_add_single(&s->map, k, 1, 1) == TPGP_OK);
    s->term_count = 3 * n;
    s->terms = calloc(s->term_count, sizeof(*s->terms));
    CHECK(s->terms != NULL);
    if (!s->terms) exit(1);
    for (size_t k = 0; k < n; k++) {
        s->terms[3*k] = (tpgp_term){k, (k+1)%n, 3, 3, .5, k};
        s->terms[3*k+1] = (tpgp_term){k, (k+1)%n, 1, 1, -.25, k};
        s->terms[3*k+2] = (tpgp_term){k, (k+1)%n, 2, 2, -.25, k};
    }
    s->options = tpgp_default_options();
    s->options.threshold = threshold;
    s->options.max_terms = 100000;
    s->options.validated = tpgp_has_mpfr();
    tpgp_ledger_init(&s->ledger, s->options.validated);
    s->ledger.peak_terms = n;
    s->dt = 1.0/64;
    s->total_steps = 4;
    s->residual_depth = tpgp_has_mpfr() ? 1 : -1;
    s->residual_bound = tpgp_has_mpfr() ? 0 : NAN;
    s->initial_global = 1;
    strcpy(s->model, "ring");
    s->j2 = .3;
    s->linear_size = 3;
}
static void advance(tpgp_checkpoint *s) {
    tpgp_map previous = {0};
    if (s->residual_depth >= 0)
        CHECK(tpgp_map_clone(&previous, &s->map) == TPGP_OK);
    CHECK(tpgp_strang_step(&s->map, s->terms, s->term_count, s->dt,
                          &s->options, &s->ledger) == TPGP_OK);
    s->pf_bound = tpgp_bound_add_up(s->pf_bound,
        tpgp_strang_bound(s->terms, s->term_count, s->dt));
    if (s->residual_depth >= 0) {
        double increment = 0;
        CHECK(tpgp_residual_bound(&previous, &s->map, s->terms, s->term_count, s->dt,
                                 (unsigned)s->residual_depth, &increment) == TPGP_OK);
        s->residual_bound = tpgp_bound_add_up(s->residual_bound, increment);
        tpgp_map_free(&previous);
    }
    s->step++;
}
static unsigned char *read_all(const char *path, size_t *length) {
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    if (!f) exit(1);
    CHECK(fseek(f, 0, SEEK_END) == 0);
    long n = ftell(f);
    CHECK(n > 0);
    rewind(f);
    unsigned char *p = malloc((size_t)n);
    CHECK(p != NULL);
    if (!p) exit(1);
    CHECK(fread(p, 1, (size_t)n, f) == (size_t)n);
    CHECK(fclose(f) == 0);
    *length = (size_t)n;
    return p;
}
static void write_all(const char *path, const unsigned char *p, size_t n) {
    FILE *f = fopen(path, "wb"); CHECK(f != NULL);
    if (!f) exit(1);
    CHECK(fwrite(p, 1, n, f) == n); CHECK(fclose(f) == 0);
}
static void test_resume(const char *path, double threshold) {
    tpgp_checkpoint continuous = {0}, interrupted = {0}, loaded = {0};
    initialize(&continuous, 4, threshold);
    initialize(&interrupted, 4, threshold);
    for (int k = 0; k < 4; k++) advance(&continuous);
    for (int k = 0; k < 2; k++) advance(&interrupted);
    char error[256];
    CHECK(tpgp_checkpoint_save(path, &interrupted, error, sizeof(error)) == TPGP_OK);
    CHECK(tpgp_checkpoint_load(path, &loaded, error, sizeof(error)) == TPGP_OK);
    CHECK(tpgp_checkpoint_same_configuration(&interrupted, &loaded));
    CHECK(equal_map(&interrupted.map, &loaded.map));
    CHECK(equal_ledger(&interrupted.ledger, &loaded.ledger));
    CHECK(equal_double(interrupted.pf_bound, loaded.pf_bound));
    CHECK(equal_double(interrupted.residual_bound, loaded.residual_bound));
    CHECK(loaded.options.metal_context == NULL && loaded.options.cuda_context == NULL);
    for (int k = 0; k < 2; k++) advance(&loaded);
    CHECK(equal_map(&continuous.map, &loaded.map));
    CHECK(equal_ledger(&continuous.ledger, &loaded.ledger));
    CHECK(equal_double(continuous.pf_bound, loaded.pf_bound));
    CHECK(equal_double(continuous.residual_bound, loaded.residual_bound));
    CHECK(loaded.step == continuous.step);
    if (threshold > 0) CHECK(loaded.ledger.dropped_l2 > 0);
    printf("restart threshold=%g: coefficients, event ledgers and cumulative bounds bitwise equal\n",
           threshold);
    tpgp_checkpoint_free(&continuous); tpgp_checkpoint_free(&interrupted);
    tpgp_checkpoint_free(&loaded);
}
static void test_rejection(const char *path, const char *bad) {
    tpgp_checkpoint original = {0}, destination = {0};
    initialize(&original, 4, 1e-5);
    char error[256];
    CHECK(tpgp_checkpoint_save(path, &original, error, sizeof(error)) == TPGP_OK);
    CHECK(tpgp_checkpoint_load(path, &destination, error, sizeof(error)) == TPGP_OK);
    size_t n;
    unsigned char *p = read_all(path, &n);
    const size_t changes[] = {0, 8, 16, 32, 64, 128, 256, 300, n/2, n-40};
    for (size_t k = 0; k < sizeof(changes)/sizeof(changes[0]); k++) {
        p[changes[k]] ^= 0x40;
        write_all(bad, p, n);
        CHECK(tpgp_checkpoint_load(bad, &destination, error, sizeof(error)) != TPGP_OK);
        CHECK(equal_map(&original.map, &destination.map));
        p[changes[k]] ^= 0x40;
    }
    p[n-1] ^= 1; write_all(bad, p, n);
    CHECK(tpgp_checkpoint_load(bad, &destination, error, sizeof(error)) != TPGP_OK);
    p[n-1] ^= 1;
    for (size_t k = 0; k < n; k += n/7) {
        write_all(bad, p, k);
        CHECK(tpgp_checkpoint_load(bad, &destination, error, sizeof(error)) != TPGP_OK);
        CHECK(equal_map(&original.map, &destination.map));
    }
    original.map.values[0] = NAN;
    CHECK(tpgp_checkpoint_save(path, &original, error, sizeof(error)) != TPGP_OK);
    size_t m; unsigned char *q = read_all(path, &m);
    CHECK(m == n && !memcmp(p, q, n));
    original.map.values[0] = 1;
    original.options.threshold = 2e-5;
    CHECK(!tpgp_checkpoint_same_configuration(&original, &destination));
    original.options.threshold = 1e-5;
    original.terms[0].h = nextafter(original.terms[0].h, INFINITY);
    CHECK(!tpgp_checkpoint_same_configuration(&original, &destination));
    free(p); free(q);
    tpgp_checkpoint_free(&original); tpgp_checkpoint_free(&destination);
    puts("corrupt/truncated checkpoints rejected transactionally; failed save preserves old bytes");
}
static void test_high_bits(const char *path) {
    tpgp_checkpoint s = {0}, r = {0};
    initialize(&s, 1000, 0);
    CHECK(tpgp_map_add_single(&s.map, 999, 3, -.625) == TPGP_OK);
    s.ledger.peak_terms = s.map.len;
    char error[256];
    CHECK(tpgp_checkpoint_save(path, &s, error, sizeof(error)) == TPGP_OK);
    CHECK(tpgp_checkpoint_load(path, &r, error, sizeof(error)) == TPGP_OK);
    CHECK(equal_map(&s.map, &r.map));
    CHECK(tpgp_checkpoint_same_configuration(&s, &r));
    tpgp_checkpoint_free(&s); tpgp_checkpoint_free(&r);
}
static void test_locked_restart(const char *program, const char *path) {
    tpgp_checkpoint s = {0};
    initialize(&s, 4, 1e-5);
    char error[256], lock_path[540];
    CHECK(tpgp_checkpoint_save(path, &s, error, sizeof(error)) == TPGP_OK);
    snprintf(lock_path, sizeof(lock_path), "%s.lock", path);
    int fd = open(lock_path, O_CREAT | O_RDWR, 0600);
    CHECK(fd >= 0);
    if (fd < 0) exit(1);
    CHECK(flock(fd, LOCK_EX | LOCK_NB) == 0);
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(fd);
        execl(program, program, "--restart", path, (char *)NULL);
        _exit(127);
    }
    if (pid > 0) {
        int status = 0;
        CHECK(waitpid(pid, &status, 0) == pid);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 2);
    }
    close(fd); unlink(lock_path);
    tpgp_checkpoint r = {0};
    CHECK(tpgp_checkpoint_load(path, &r, error, sizeof(error)) == TPGP_OK);
    CHECK(equal_map(&s.map, &r.map) && s.step == r.step);
    tpgp_checkpoint_free(&s); tpgp_checkpoint_free(&r);
    puts("concurrent restart rejected while project checkpoint lock held");
}
int main(int argc, char **argv) {
    char dir[] = "/tmp/tpgp_checkpoint_rev_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char path[512], bad[512], manifest[540];
    snprintf(path, sizeof(path), "%s/state.bin", dir);
    snprintf(bad, sizeof(bad), "%s/corrupt.bin", dir);
    snprintf(manifest, sizeof(manifest), "%s.manifest.json", path);
    test_resume(path, 0); test_resume(path, 1e-5);
    test_rejection(path, bad); test_high_bits(path);
    if (argc == 2) test_locked_restart(argv[1], path);
    unlink(path); unlink(bad); unlink(manifest); rmdir(dir);
    printf("checkpoint checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
