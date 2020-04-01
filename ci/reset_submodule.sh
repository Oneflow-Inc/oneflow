set -x
set -e
git reset --hard
git submodule deinit -f .
rm -rf .git/modules/*
