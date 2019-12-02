#include "oneflow/core/framework/blob.h"

namespace oneflow {

namespace user_op {

Blob::Blob(const BlobDef& def, char* dptr) : def_(def), dptr_(dptr) {}

Blob::Blob(const Shape& shape, DataType dtype, char* dptr) : def_(shape, dtype), dptr_(dptr) {}

Blob::Blob(const Blob& rhs) {
  def_ = rhs.def_;
  dptr_ = rhs.dptr_;
}

}  // namespace user_op

}  // namespace oneflow
