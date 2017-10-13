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

#define FLAXIBLE_STRUCT_SEQ                                              \
  OF_PP_MAKE_TUPLE_SEQ(DataSetFeatureHeader, dim_array_size, dim_vec)    \
  OF_PP_MAKE_TUPLE_SEQ(DataItem, len, data)                              \
  OF_PP_MAKE_TUPLE_SEQ(DataSetLabelHeader, label_array_size, label_name) \
  OF_PP_MAKE_TUPLE_SEQ(DataSetLabel, len, data_item_label_idx)

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
};

struct DataSetFeatureHeader final {
  uint32_t dim_array_size = 0;
  uint32_t dim_vec[0];  //  shape

  size_t ElementCount() const {
    int count = 1;
    for (int i = 0; i < dim_array_size; i++) { count *= dim_vec[i]; }
    return count;
  }
};

struct DataSetLabelHeader final {
  uint32_t label_array_size = 0;
  char label_name[0][64];  // label dicription
};

struct DataItem final {
  uint64_t len = 0;  //  len = dim_vec[0] * dev_vec[1] * ...
  double data[0];    //	tensor data.
};

struct DataSetLabel final {
  uint64_t len = 0;                 //  len = data_item_size
  uint32_t data_item_label_idx[0];  //	label of data item.
};

template<typename data_set_class>
size_t FlexibleSizeOf(uint32_t n);

template<typename data_set_class>
size_t FlexibleSizeOf(const data_set_class& obj);

#define DATA_SET_DECLARE_OFSTREAM(type) \
  std::ostream& operator<<(std::ostream& out, const type& data);
OF_PP_FOR_EACH_TUPLE(DATA_SET_DECLARE_OFSTREAM, DATA_SET_FORMAT_SEQ);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
