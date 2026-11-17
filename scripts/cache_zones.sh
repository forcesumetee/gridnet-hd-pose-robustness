#!/usr/bin/env bash
# Download + cache zones one at a time. Skips zones whose report.txt already exists.
#   bash scripts/cache_zones.sh t1z5b t1z8 t3z4 t4z1 t5b1
set -u
mkdir -p logs
for z in "$@"; do
  if [ -f "data/cache/$z/report.txt" ]; then echo "[skip] $z done"; continue; fi
  echo "[$(date '+%H:%M')] downloading $z"
  hf download heig-vd-geo/GridNet-HD --repo-type dataset --include "$z/*" --local-dir ./data > "logs/dl_$z.log" 2>&1 \
    || { echo "[fail] download $z (see logs/dl_$z.log)"; continue; }
  echo "[$(date '+%H:%M')] caching $z"
  python scripts/build_cache.py --zone "data/$z" --out data/cache --voxel 0.05 --k 8 > "logs/cache_$z.log" 2>&1 \
    && echo "[$(date '+%H:%M')] done $z" || echo "[fail] cache $z (see logs/cache_$z.log)"
done
echo "all finished $(date '+%H:%M')"
