#!/bin/bash
set -euo pipefail
B=$1;mkdir "$2";cd "$2"
rc=0
TPGP_ORBIT_BUCKETS=1 TPGP_ORBIT_SAVE=prior.ck "$B" 8 .03125 0 2 5000 1200 > first.csv 2>first.log || rc=$?
[[ $rc == 71 && -s prior.ck ]]
TPGP_ORBIT_BUCKETS=4 TPGP_ORBIT_SAVE=resumed.ck TPGP_ORBIT_RESTART=prior.ck "$B" 8 .03125 0 2 5000 1200 >recovered.csv 2>recovered.log
TPGP_ORBIT_BUCKETS=1 TPGP_ORBIT_SAVE=ordinary.ck "$B" 8 .03125 0 2 5000 5000 >ordinary.csv 2>ordinary.log
"$(dirname "$B")/bucket_verify" recovery
