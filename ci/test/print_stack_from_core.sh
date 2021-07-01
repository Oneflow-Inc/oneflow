set -ex
if compgen -G "$2/core.*" > /dev/null; then
    gdb --batch --quiet -ex "thread apply all bt full" -ex "quit" $1 $2/core.*
fi
