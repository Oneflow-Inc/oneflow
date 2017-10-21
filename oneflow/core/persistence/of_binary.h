#ifndef ONEFLOW_CORE_PERSISTENCE_RECORD_H_
#define ONEFLOW_CORE_PERSISTENCE_RECORD_H_
#include <cstdint>
#include <iostream>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/flexible.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/persistence/ofb_decoder.h"
#include "oneflow/core/persistence/persistent_out_stream.h"
namespace oneflow {

//	data set format
//	.-------------------------.
//	| OfbHeader | OfbItem ... |
//	'-------------------------'

#define DATA_SET_FORMAT_SEQ       \
  OF_PP_MAKE_TUPLE_SEQ(OfbHeader) \
  OF_PP_MAKE_TUPLE_SEQ(OfbItem)

//  oneflow binary file header
struct OfbHeader final {
  const uint16_t magic_code_ = 0xfeed;
  const uint16_t version_ = 0;
  uint32_t check_sum_;            // check header
  char type_[16];                 //  "feature" or "label"
  uint32_t dim_array_size_ = 0;   //  effective length of dim_array
  uint32_t dim_array_[15];        //  tensor shape
  uint64_t data_item_count_ = 0;  //  how many items after header

  OF_DISALLOW_COPY_AND_MOVE(OfbHeader);
  OfbHeader() = default;
};

//  oneflow binary file item
//  OfbItem is basically a key-value pair
class OfbItem final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfbItem);
  OfbItem() = delete;
  uint8_t meta_check_sum_;  //	check fields except `data'
  uint8_t data_check_sum_;  //  checking `data' field when debugging
  uint8_t data_type_ = DataType::kChar;                   // value data type
  uint8_t data_encode_type_ = DataEncodeType::kNoEncode;  // value encode type
  uint32_t len_ = 0;      //  len = flexible sizeof(data) / sizeof(data[0])
  uint16_t key_len_ = 0;  // key string length
  uint16_t value_offset_ =
      0;  //  value data offset in field `data', which is >=
          //  key_len and 8-byte aligned
  uint32_t _8_byte_alignment_ = 0;  //  useless, only for alignment

  //  data layout:
  //  |<------- value_offset ------->|
  //  |<------- key_len ------>|     |
  //	+------------------------+-----+-----------------------.
  //	| key data (string type) | \0* | value data (any type) |
  //	'------------------------------------------------------'
  char data_[0];  //  key string data + value data.

  std::string GetDataId() const;

  template<typename T>
  void Decode(const Shape& shape, T* out_dptr);

  size_t key_buffer_len() const { return key_len_; }
  const char* key_buffer() const { return data_; }
  size_t value_buffer_len() const { return len_ - value_offset_; }
  const char* value_buffer() const { return data_ + value_offset_; }

  char* mut_key_buffer() { return data_; }
  char* mut_value_buffer() { return data_ + value_offset_; }
};

#define DECLARE_DATA_SET_OFSTREAM(type) \
  std::ofstream& operator<<(std::ofstream& out, const type& data);
OF_PP_FOR_EACH_TUPLE(DECLARE_DATA_SET_OFSTREAM, DATA_SET_FORMAT_SEQ);

#define DECLARE_DATA_SET_PERSISTENCE_OUT(type) \
  PersistentOutStream& operator<<(PersistentOutStream& out, const type& data);
OF_PP_FOR_EACH_TUPLE(DECLARE_DATA_SET_PERSISTENCE_OUT, DATA_SET_FORMAT_SEQ);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_RECORD_H_
