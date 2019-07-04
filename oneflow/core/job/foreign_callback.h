#ifndef ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
#define ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_

#include "oneflow/core/register/ofblob.h"

namespace oneflow {

class ForeignCallback {
 public:
  ForeignCallback() = default;

  virtual ~ForeignCallback() = default;

  virtual void PushBlob(uint64_t ofblob_ptr) const { UNIMPLEMENTED(); }
  virtual void PullBlob(uint64_t ofblob_ptr) const { UNIMPLEMENTED(); }
  virtual void Finish() const { UNIMPLEMENTED(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
