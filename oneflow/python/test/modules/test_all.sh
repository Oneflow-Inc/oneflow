ls |grep 'test.*.py' > test_cases_names.txt
while read line
do
    echo "Runing test >>>>>>>>>>>> " $line
    ONEFLOW_TEST_ENABLE_EAGER=1 python3 $line
    sleep 1
done < test_cases_names.txt
