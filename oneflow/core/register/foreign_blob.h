#ifndef ONEFLOW_CORE_REGISTER_FOREIGN_BLOB_H_
#define ONEFLOW_CORE_REGISTER_FOREIGN_BLOB_H_

#include "oneflow/core/register/blob.h"

namespace oneflow {

class ForeignBlob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignBlob);
  virtual ~ForeignBlob() = default;

  virtual void CopyFrom(const Blob* blob) = 0;
  virtual void CopyTo(Blob* blob) const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_FOREIGN_BLOB_H_
