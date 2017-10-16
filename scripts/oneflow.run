set -e
sed -e '1,/^exit$/d' "$0" | tar zxf -
./scheduler $@
exit
