#ifndef ONEFLOW_CORE_PERSISTENCE_UBF_ITEM_H_
#define ONEFLOW_CORE_PERSISTENCE_UBF_ITEM_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/flexible.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

#define DATA_ENCODE_TYPE_SEQ      \
  OF_PP_MAKE_TUPLE_SEQ(kNoEncode) \
  OF_PP_MAKE_TUPLE_SEQ(kJpeg)     \
  OF_PP_MAKE_TUPLE_SEQ(kSparse)

enum DataEncodeType {
#define DECLARE_DATA_ENCODE_TYPE(encode_type) encode_type,
  OF_PP_FOR_EACH_TUPLE(DECLARE_DATA_ENCODE_TYPE, DATA_ENCODE_TYPE_SEQ)
};

//	binary format
//	.-------------------------.
//	| UbfHeader | UbfItem ... |
//	'-------------------------'

//  unified binary formatted item
//  UbfItem is basically a key-value pair
class UbfItem final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfItem);
  UbfItem() = delete;

  static std::unique_ptr<UbfItem, decltype(&free)> New(
      const std::string& key, size_t value_buf_len, DataType dtype,
      DataEncodeType detype, const std::function<void(char* buff)>& Fill);
  static std::unique_ptr<UbfItem, decltype(&free)> NewEmpty() {
    return Flexible<UbfItem>::Malloc(0);
  }

  std::string GetDataId() const;
  template<typename T>
  void Decode(const Shape& shape, T* out_dptr) const;

  // add all bytes except `data` field one by one
  uint8_t ComputeMetaCheckSum() const;

  //	getter
  DataType data_type() const { return static_cast<DataType>(data_type_); }
  DataEncodeType data_encode_type() const {
    return static_cast<DataEncodeType>(data_encode_type_);
  }
  size_t len() const { return len_; }
  size_t key_buffer_len() const { return key_len_; }
  const char* key_buffer() const { return data_; }
  size_t value_buffer_len() const { return len_ - value_offset_; }
  const char* value_buffer() const { return data_ + value_offset_; }

  //	setter
  char* mut_data() { return data_; }
  char* mut_key_buffer() { return data_; }
  char* mut_value_buffer() { return data_ + value_offset_; }

  friend class Flexible<UbfItem>;

 private:
  // add all bytes of `data` field one by one
  uint8_t ComputeDataCheckSum() const;
  void UpdateCheckSum();
  void UpdateDataCheckSum();
  void UpdateMetaCheckSum();

  uint8_t meta_check_sum_;               //	check fields except `data'
  uint8_t data_check_sum_;               //  check `data' field when debugging
  uint8_t data_type_ = DataType::kChar;  // value data type
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
};

PersistentOutStream& operator<<(PersistentOutStream& out, const UbfItem& data);

template<DataEncodeType encode_type>
class UbfDecoder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfDecoder);
  UbfDecoder() = delete;
  template<typename T>
  static void Decode(const UbfItem& ubf_item, const Shape& shape, T* out_dptr);

 private:
  template<typename src_type, typename T>
  static void Cast(const UbfItem& ubf_item, const Shape& shape, T* out_dptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_UBF_ITEM_H_
