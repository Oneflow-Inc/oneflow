#ifndef ONEFLOW_CORE_PERSISTENCE_UBF_ITEM_H_
#define ONEFLOW_CORE_PERSISTENCE_UBF_ITEM_H_

#include "oneflow/core/common/data_type.h"
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

class UbfItemDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfItemDesc);
  UbfItemDesc() = default;
  ~UbfItemDesc() = default;
  UbfItemDesc(DataType dtype, DataEncodeType detype, size_t data_id_len,
              size_t body_len)
      : data_type_(dtype),
        data_encode_type_(detype),
        data_id_len_(data_id_len),
        len_(ComputeLength(data_id_len, body_len)) {}

  static size_t ComputeLength(size_t data_id_len, size_t body_len) {
    return RoundUpToAlignment(data_id_len, 8) + body_len;
  }

  //	getter
  DataType data_type() const { return static_cast<DataType>(data_type_); }
  DataEncodeType data_encode_type() const {
    return static_cast<DataEncodeType>(data_encode_type_);
  }
  size_t len() const { return len_; }
  size_t data_id_len() const { return data_id_len_; }
  size_t body_offset() const { return RoundUpToAlignment(data_id_len_, 8); }
  size_t body_len() const { return len() - body_offset(); }

  //	setter
  void set_body_len(size_t body_len) { len_ = body_offset() + body_len; }

 private:
  uint8_t data_type_ = DataType::kChar;                   // value data type
  uint8_t data_encode_type_ = DataEncodeType::kNoEncode;  // value encode type
  uint16_t data_id_len_ = 0;                              // key string length
  uint32_t len_ = 0;                                      //
  //  data layout:
  //  |<---------------------------- len_ ------------------------------>|
  //  |<-- RoundUpToAlignment(data_id_len_, 8)-->|                       |
  //  |<---------- data_id_len_ ---------->|     |                       |
  //	+------------------------------------+-----+-----------------------+
  //	|        data id (string type)       | \0* |    body (any type)    |
  //	'------------------------------------------------------------------'
};

//  united binary format
//	.-------------------------.
//	| UbfHeader | UbfItem ... |
//	'-------------------------'

//  united binary formatted item
//  UbfItem layout
//  .------------------------------.
//  | UbfItemDesc | data_id + body |
//  '------------------------------'
//  UbfItem is basically a key-value pair
class UbfItem final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfItem);
  UbfItem() = delete;
  explicit UbfItem(std::unique_ptr<UbfItemDesc>&& desc)
      : desc_(std::move(desc)),
        data_(std::unique_ptr<char, decltype(&free)>(
            static_cast<char*>(malloc(desc_->len())), &free)) {}
  UbfItem(DataType dtype, DataEncodeType detype, const std::string& data_id,
          size_t body_len, const std::function<void(char*)>& Fill);

  template<typename T>
  void Decode(const Shape& shape, T* out_dptr) const;

  //	getter
  const UbfItemDesc* desc() const { return desc_.get(); }
  DataType data_type() const { return desc()->data_type(); }
  DataEncodeType data_encode_type() const { return desc()->data_encode_type(); }
  size_t len() const { return desc()->len(); }
  const char* data() const { return data_.get(); }
  std::string data_id() const { return std::string(data(), data_id_len()); }
  const char* body() const { return data() + desc()->body_offset(); }
  size_t body_len() const { return desc()->body_len(); }

  //	setter
  char* mut_data() { return data_.get(); }

 private:
  size_t data_id_len() const { return desc()->data_id_len(); }
  char* mut_data_id() { return data_.get(); }
  char* mut_body() { return mut_data() + desc()->body_offset(); }

  std::unique_ptr<UbfItemDesc> desc_;
  std::unique_ptr<char, decltype(&free)> data_;
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
