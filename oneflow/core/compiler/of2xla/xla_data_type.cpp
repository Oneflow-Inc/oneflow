#include "glog/logging.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/compiler/of2xla/xla_data_type.h"

namespace oneflow {
namespace mola {

xla::PrimitiveType DataTypeToPrimitiveType(DataType data_type) {
  switch (data_type) { 
    case oneflow::kFloat:
      return xla::F32;
    case oneflow::kDouble:
      return xla::F64;
    case oneflow::kInt8:
      return xla::S8;
    case oneflow::kInt32:
      return xla::S32;
    case oneflow::kInt64:
      return xla::S64;
    case oneflow::kChar:
    case oneflow::kUInt8:
      return xla::U8;
    default:
      LOG(FATAL) << "Unsupported data type (" << data_type
                 << ") in DataTypeToPrimitiveType";
      return xla::PRIMITIVE_TYPE_INVALID;
  }
}

}  // namespace mola
}  // namespace oneflow
