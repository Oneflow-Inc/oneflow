#!/bin/bash

python test_gather.py &> /dev/null
wait
echo $?
