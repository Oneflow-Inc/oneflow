#include "oneflow/core/framework/blob.h"

namespace oneflow {

namespace user_op {

Blob::Blob(const BlobDef& def, char* dptr) : def_(def), dptr_(dptr) {}
Blob::Blob(const Shape& shape, DataType dtype, char* dptr) : def_(shape, dtype), dptr_(dptr) {}

}  // namespace user_op

}  // namespace oneflow
