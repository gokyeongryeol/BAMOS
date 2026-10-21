#!/usr/bin/env bash
# A data root for habitat_server/teleport_server.py --data-root: data/datasets from the objectnav data, and
# data/scene_datasets/hssd-hab pointing at the hssd-hab v0.2.4 copy (config files at tag v0.2.4, GLBs linked to the
# identical v0.2.5 files). The ObjectNav episodes were generated on v0.2.4 assets.
#
#   bash scripts/make_hssd_root.sh [root] [objectnav data dir] [v0.2.4 copy]
set -eu
cd "$(dirname "$0")/.."
ROOT=${1:-results/cache/hssd_v024/root}
DATA=${2:-${OBJECTNAV_DATA:?set OBJECTNAV_DATA to the habitat data directory (datasets/, scene_datasets/)}}
V024=${3:-$DATA/scene_datasets/hssd-hab-v0.2.4-meta}
mkdir -p "$ROOT/data/scene_datasets"
ln -sfn "$DATA/datasets" "$ROOT/data/datasets"
ln -sfn "$V024" "$ROOT/data/scene_datasets/hssd-hab"
echo "$ROOT: datasets -> $DATA/datasets, hssd-hab -> $V024"
