#!/bin/bash
# Run this script at project root by "./ci/linter_py.sh" before you commit

set -e

vergte() {
  [ "$2" = "$(echo -e "$1\\n$2" | sort -V | head -n1)" ]
}

{
	black --version | grep "19.1" > /dev/null
} || {
	echo "Linter requires black>=19.1 !"
	exit 1
}

ISORT_TARGET_VERSION="4.3.21"
ISORT_VERSION=$(isort -v | grep VERSION | awk '{print $2}')
vergte "$ISORT_VERSION" "$ISORT_TARGET_VERSION" || {
  echo "Linter requires isort>=${ISORT_TARGET_VERSION} !"
  exit 1
}

{
	flake8 --version | grep "3.8.2" > /dev/null
} || {
	echo "Linter requires flake8>=3.8.2 !"
	exit 1
}

if [ "$*" ]; then
    current_path_or_file="$*";
else
    current_path=$(readlink -f "$(dirname "$0")");
    cd $current_path
    cd ../oneflow/python
    current_path_or_file=$(cd "$(dirname $0)";pwd)
fi

echo "Running isort ..."
isort -rc --atomic $current_path_or_file

echo "Running black ..."
black -l 79 $current_path_or_file

echo "Running flake8 ..."

flake8 --ignore=E231,E731,E203,E741,W503,W293,W605,E501 --format="%(path)s:%(row)d:%(col)d  |  %(code)s  |  %(text)s" $current_path_or_file


if [[ "$?" == 0 ]]; then
  echo "Ok!"
  exit 0
else
  echo "Check failed！"
  exit 1
fi
