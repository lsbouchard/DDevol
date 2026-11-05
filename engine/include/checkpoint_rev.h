#ifndef CHECKPOINT_REV_H
#define CHECKPOINT_REV_H
#include "tpgp_rev.h"

/* Owned restart record. Zero-initialize before load/free. Context pointers in
   options are runtime-only and are never serialized. */
typedef struct {
    tpgp_map map;
    tpgp_term *terms;
    size_t term_count;
    tpgp_options options;
    tpgp_ledger ledger;
    double dt, pf_bound, residual_bound, max_fid_bound;
    size_t step, total_steps;
    int residual_depth; /* -1 means not requested; bound is then NaN */
    int initial_global;
    char model[32];
    double j2;
    size_t linear_size;
    size_t resumed_from_step, prior_total_steps, prior_max_terms;
    int stop_reason; /* 0 running,1 complete,2 walltime,3 signal,4 resource,5 numerical,6 accuracy */
    tpgp_status last_status;
} tpgp_checkpoint;

void tpgp_checkpoint_free(tpgp_checkpoint *checkpoint);
/* Save writes a little-endian version-2 record plus SHA-256 checksum through
   an exclusive temporary file, fflush/fsync, and atomic rename. A companion
   .manifest.json contains the payload hash and human-readable run metadata.
   A rename is durable only after the parent-directory fsync also succeeds. */
tpgp_status tpgp_checkpoint_save(const char *path, const tpgp_checkpoint *checkpoint, char *error,
                                 size_t error_capacity);
/* Verify the complete checksum before allocating numerical arrays, parse
   strictly, and replace dst only after every invariant passes. */
tpgp_status tpgp_checkpoint_load(const char *path, tpgp_checkpoint *dst, char *error,
                                 size_t error_capacity);
/* Exact run-configuration comparison, excluding state, current step, runtime
   contexts, output paths, accuracy-stop threshold and checkpoint/walltime
   cadence. Includes total_steps/max_terms; the CLI separately permits explicit
   increases to these two resource/horizon settings on restart. */
int tpgp_checkpoint_same_configuration(const tpgp_checkpoint *a, const tpgp_checkpoint *b);
#endif
