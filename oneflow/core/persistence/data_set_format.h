#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
#include <cstdint>
#include <iostream>
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"
namespace oneflow {

//	data set format
//
//	for feature file
//	.------------------------------.
//	| DataSetHeader | DataItem ... |
//	'------------------------------'
//
//	for label file
//	.-------------------------------------------------.
//	| DataSetHeader | DataSetLabelDesc | DataItem ... |
//	'-------------------------------------------------'

#define FLAXIBLE_STRUCT_SEQ                                            \
  OF_PP_MAKE_TUPLE_SEQ(DataSetLabelDesc, label_array_size, label_desc) \
  OF_PP_MAKE_TUPLE_SEQ(DataItem, len, data)

#define DATA_SET_FORMAT_SEQ              \
  OF_PP_MAKE_TUPLE_SEQ(DataSetHeader)    \
  OF_PP_MAKE_TUPLE_SEQ(DataSetLabelDesc) \
  OF_PP_MAKE_TUPLE_SEQ(DataItem)

struct DataSetHeader final {
  const uint32_t magic_code = 0xfeed;
  const uint32_t version = 0;
  char type[16];                    //  "feature" or "label"
  uint32_t label_desc_buf_len = 0;  //  in bytes, only for label
  uint16_t data_elem_type = 0;      // type of data element
  uint16_t dim_array_size = 0;      // effective length of dim_array
  uint32_t dim_array[16];           //  tensor shape
  uint64_t data_item_count = 0;     //  how many items after header

  OF_DISALLOW_COPY_AND_MOVE(DataSetHeader);
  DataSetHeader() = default;
  size_t TensorElemCount() const {
    int count = 1;
    for (int i = 0; i < dim_array_size; i++) { count *= dim_array[i]; }
    return count;
  }
  size_t DataBodyOffset() const { return sizeof(*this) + label_desc_buf_len; }
};

struct DataSetLabelDesc final {
  uint32_t label_array_size = 0;
  char label_desc[0][128];  // label dicription

  OF_DISALLOW_COPY_AND_MOVE(DataSetLabelDesc);
  DataSetLabelDesc() = delete;
};

struct DataItem final {
  uint64_t len = 0;  //  len = dim_vec[0] * dev_vec[1] * ...
  uint32_t data[0];  //	tensor data.

  OF_DISALLOW_COPY_AND_MOVE(DataItem);
  DataItem() = delete;
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
