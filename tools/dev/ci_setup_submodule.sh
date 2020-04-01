set -x
set -e
git reset --hard
git submodule deinit -f .
rm -rf .git/modules/*
python3 tools/dev/setup_submodule.py --oneflow_src_local_path=${ONEFLOW_CI_SRC_DIR}
git submodule sync
git submodule update --init --recursive
