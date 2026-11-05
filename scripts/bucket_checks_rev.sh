#!/bin/bash
set -euo pipefail
B=$1;D=$2;dense=$3;mkdir "$D";cd "$D"
for n in 8 64;do for eta in 0 1e-7;do
 k=2; [[ $eta != 0 || $n != 64 ]] || k=1
 for q in 1 4;do
 tag="${n}_${eta}_${q}";TPGP_ORBIT_BUCKETS=$q TPGP_ORBIT_SAVE="$tag.ck" "$B" "$n" .03125 "$eta" "$k" 1000000 24000000 >"$tag.csv" 2>"$tag.log"
 done;done;done
export TPGP_ORBIT_BUCKETS=4
for k in 4 8;do TPGP_ORBIT_SAVE="s$k.ck" "$B" 8 .03125 1e-7 "$k" >"s$k.csv" 2>"s$k.log";done
TPGP_ORBIT_RESTART=s4.ck TPGP_ORBIT_SAVE=restart.ck "$B" 8 .03125 1e-7 8 >resume.csv 2>resume.log
cmp s8.ck restart.ck
cp s4.ck bad.ck;printf 'X' |dd of=bad.ck bs=1 seek=12 conv=notrunc status=none
head -c 80 s4.ck > trunc.ck
for bad in bad.ck trunc.ck;do if TPGP_ORBIT_RESTART=$bad "$B" 8 .03125 1e-7 8 >/dev/null 2>&1;then exit 1;fi;done
if TPGP_ORBIT_RESTART=s4.ck "$B" 8 .015625 1e-7 8 >/dev/null 2>&1;then exit 1;fi
"$B" 8 .015625 0 104 >dense_compare.csv 2>dense_compare.log
"$(dirname "$B")/bucket_verify" suite "$dense"
