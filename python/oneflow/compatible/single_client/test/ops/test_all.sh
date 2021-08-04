# Test all ops, usage :
# test 1n2g: bash test_all.sh > test_1_gpu.log 2>&1
# test 1n2g: export ONEFLOW_TEST_DEVICE_NUM=2 && bash test_all.sh > test_2_gpu.log 2>&1
ls |grep 'test.*.py' > test_cases_names.txt
while read line
do
    echo "Runing test >>>>>>>>>>>> " $line
    python3 $line
    sleep 1
done < test_cases_names.txt
rm test_cases_names.txt

