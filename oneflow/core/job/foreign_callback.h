#ifndef ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
#define ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_

#include "oneflow/core/register/foreign_blob.h"

namespace oneflow {

class ForeignCallback {
 public:
  ForeignCallback() = default;

  virtual ~ForeignCallback() = default;

  virtual void Run(const ForeignBlob&) const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
