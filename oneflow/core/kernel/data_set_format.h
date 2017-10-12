#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
#include <cstdint>
#include <iostream>
#include "oneflow/core/common/preprocessor.h"
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

#define DATA_SET_FORMAT_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(DataSetHeaderDesc)    \
  OF_PP_MAKE_TUPLE_SEQ(DataSetFeatureHeader) \
  OF_PP_MAKE_TUPLE_SEQ(DataItem)             \
  OF_PP_MAKE_TUPLE_SEQ(DataSetLabelHeader)   \
  OF_PP_MAKE_TUPLE_SEQ(DataSetLabel)

struct DataSetHeaderDesc final {
  const uint32_t magic_code = 0xfeed;
  const uint32_t version = 0;
  char type[12];               //  "feature" or "label"
  uint32_t header_buffer_len;  //  in bytes
  uint32_t data_item_size;     //  how many items after header

  size_t Size() const { return sizeof(*this); }
};

struct DataSetFeatureHeader final {
  uint32_t dim_array_size = 0;
  uint32_t dim_vec[0];  //  shape

  size_t Size() const {
    return sizeof(dim_array_size) + dim_array_size * sizeof(dim_vec[0]);
  }

  size_t ElementCount() const {
    int count = 1;
    for (int i = 0; i < dim_array_size; i++) { count *= dim_vec[i]; }
    return count;
  }
};

struct DataSetLabelHeader final {
  uint32_t label_array_size;
  char label_name[0][64];  // label dicription

  size_t Size() const {
    return sizeof(label_array_size) + label_array_size * sizeof(label_name[0]);
  }
};

struct DataItem final {
  uint64_t len;    //  len = dim_vec[0] * dev_vec[1] * ...
  double data[0];  //	tensor data.

  size_t Size() const { return sizeof(len) + len * sizeof(data[0]); }
};

struct DataSetLabel final {
  uint64_t len;                     //  len = data_item_size
  uint32_t data_item_label_idx[0];  //	label of data item.

  size_t Size() const {
    return sizeof(len) + len * sizeof(data_item_label_idx[0]);
  }
};

#define DATA_SET_DECLARE_OFSTREAM(type) \
  std::ostream& operator<<(std::ostream& out, const type& data);
OF_PP_FOR_EACH_TUPLE(DATA_SET_DECLARE_OFSTREAM, DATA_SET_FORMAT_SEQ);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
