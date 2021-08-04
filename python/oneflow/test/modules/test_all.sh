Test all ops
ls |grep 'test.*.py' > test_cases_names.txt
while read line
do
    echo "Runing test >>>>>>>>>>>> " $line
    python3 $line -v
    sleep 1
done < test_cases_names.txt
rm test_cases_names.txt

