set -ex
find . -type f -name "core.*" -exec gdb --batch --quiet -ex "thread apply all bt full" -ex "quit" python3 {} \;
