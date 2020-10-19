set -x
set -e
src_dir=${ONEFLOW_CI_SRC_DIR:-"$HOME/oneflow"}
python3 ci/setup_submodule.py --oneflow_src_local_path=$src_dir
git submodule sync
git submodule update --init --recursive
