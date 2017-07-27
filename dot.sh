for dot_file in `find . -name *.dot`
do
  echo "process $dot_file"
  dot -Tpng -O $dot_file && rm $dot_file
done
