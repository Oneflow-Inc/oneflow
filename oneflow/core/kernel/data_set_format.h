#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
namespace oneflow {

//	data set format
//
//	for feature file
//	.---------------------------------------------------------.
//	| DataSetHeaderDesc | DataSetFeatureHeader | DataItem ... |
//	'---------------------------------------------------------'
//
//	for label file
//	.-------------------------------------------------------.
//	| DataSetHeaderDesc | DataSetLabelHeader | DataSetLabel |
//	'-------------------------------------------------------'

struct DataSetHeaderDesc final {
  int32_t version = 0;
  char type[12];              //  "feature" or "label"
  int32_t header_buffer_len;  //  in bytes
  int32_t data_item_size;     //  how many items after header
};

struct DataSetFeatureHeader final {
  int32_t dim_array_size;
  int32_t dim_vec[0];  //  shape
};

struct DataSetLabelHeader final {
  int32_t label_array_size;
  char label_name[0][64];  // label dicription
};

struct DataItem final {
  double data[0];  //	tensor data. len = dim_vec[0] * dev_vec[1] * ...
};

struct DataSetLabel final {
  int32_t data_item_label_idx[0];  //	label of data item. len = data_item_size
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
