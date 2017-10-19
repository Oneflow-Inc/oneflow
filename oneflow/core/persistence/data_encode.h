#ifndef ONEFLOW_CORE_PERSISTENCE_DATA_ENCODE_H_
#define ONEFLOW_CORE_PERSISTENCE_DATA_ENCODE_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define DATA_ENCODE_TYPE_SEQ      \
  OF_PP_MAKE_TUPLE_SEQ(kNoEncode) \
  OF_PP_MAKE_TUPLE_SEQ(kJpeg)     \
  OF_PP_MAKE_TUPLE_SEQ(kSparse)

enum DataEncodeType {
#define DECLARE_DATA_ENCODE_TYPE(encode_type) encode_type,
  OF_PP_FOR_EACH_TUPLE(DECLARE_DATA_ENCODE_TYPE, DATA_ENCODE_TYPE_SEQ)
};

class Record;
template<DataEncodeType encode_type>
class RecordDecoder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordDecoder);
  RecordDecoder() = delete;
  template<typename T>
  static void Decode(const Record& record, const Shape& shape, T* out_dptr);
  template<typename src_type, typename T>
  static void Cast(const Record& record, const Shape& shape, T* out_dptr);
};
}
#endif  // ONEFLOW_CORE_PERSISTENCE_DATA_ENCODE_H_
