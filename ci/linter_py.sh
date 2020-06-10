#!/bin/bash -e

# Run this script at project root by "./ci/linter_py.sh" before you commit

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

set -v
file_or_directory="."
python_dir="oneflow/python"

if [ "$*" ];
then
    file_or_directory="$*";
else
    current_path=$(readlink -f "$(dirname "$0")");
    cd $current_path
    cd ..
    cd $python_dir
fi

echo "Running isort ..."
isort -rc --atomic "$file_or_directory"

echo "Running black ..."
black -l 79 "$file_or_directory"

echo "Running flake8 ..."
flake8 --ignore=E231,E731,E203,E741,W503,E501,W293,W605 "$file_or_directory"

command -v arc > /dev/null && arc lint
