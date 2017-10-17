#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
#include <cstdint>
#include <iostream>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/persistent_out_stream.h"
namespace oneflow {

//	data set format
//	.----------------------------.
//	| DataSetHeader | Buffer ... |
//	'----------------------------'

#define FLAXIBLE_STRUCT_SEQ OF_PP_MAKE_TUPLE_SEQ(Buffer, len, data)

#define DATA_SET_FORMAT_SEQ           \
  OF_PP_MAKE_TUPLE_SEQ(DataSetHeader) \
  OF_PP_MAKE_TUPLE_SEQ(Buffer)

enum DataCompressType {
  kNoCompress,
  kJpeg,
  kSparse,
};

struct DataSetHeader final {
  const uint16_t magic_code = 0xfeed;
  const uint16_t version = 0;
  uint32_t check_sum;            // check header
  char type[16];                 //  "feature" or "label"
  uint32_t dim_array_size = 0;   //  effective length of dim_array
  uint32_t dim_array[15];        //  tensor shape
  uint64_t data_item_count = 0;  //  how many items after header

  OF_DISALLOW_COPY_AND_MOVE(DataSetHeader);
  DataSetHeader() = default;
  size_t TensorElemCount() const;
  size_t DataBodyOffset() const;
};

struct Buffer final {
  uint8_t meta_check_sum;  //	check fields except `data'
  uint8_t data_check_sum;  //  checking `data' field when debugging
  uint8_t data_type = DataType::kChar;
  uint8_t data_compress_type = DataCompressType::kNoCompress;
  uint32_t align = 0;  //  useless, only for alignment
  uint64_t len = 0;    //  len = sizeof(data) / sizeof(data[0])
  char data[0];        //  buffer data.

  OF_DISALLOW_COPY_AND_MOVE(Buffer);
  Buffer() = delete;
};

template<typename flexible_struct>
size_t FlexibleSizeOf(uint32_t n) {
  return sizeof(flexible_struct);
}

template<typename flexible_struct>
size_t FlexibleSizeOf(const flexible_struct& obj) {
  return sizeof(flexible_struct);
}

template<typename flexible_struct>
void FlexibleSetArraySize(flexible_struct* type, size_t len) {}

template<typename T>
static std::unique_ptr<T, decltype(&free)> FlexibleMalloc(size_t len) {
  T* ptr = reinterpret_cast<T*>(malloc(FlexibleSizeOf<T>(len)));
  FlexibleSetArraySize(ptr, len);
  return std::unique_ptr<T, decltype(&free)>(ptr, &free);
}

#define DECLARE_DATA_SET_OFSTREAM(type) \
  std::ofstream& operator<<(std::ofstream& out, const type& data);
OF_PP_FOR_EACH_TUPLE(DECLARE_DATA_SET_OFSTREAM, DATA_SET_FORMAT_SEQ);

#define DECLARE_DATA_SET_PERSISTENCE_OUT(type) \
  PersistentOutStream& operator<<(PersistentOutStream& out, const type& data);
OF_PP_FOR_EACH_TUPLE(DECLARE_DATA_SET_PERSISTENCE_OUT, DATA_SET_FORMAT_SEQ);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_FORMAT_H_
