#ifndef ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
#define ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_

#include "oneflow/core/register/ofblob.h"

namespace oneflow {

class ForeignCallback {
 public:
  ForeignCallback() = default;

  virtual ~ForeignCallback() = default;

  virtual std::string job_name() const { UNIMPLEMENTED(); }
  virtual std::string sole_input_op_name_in_user_job() const { UNIMPLEMENTED(); }
  virtual std::string sole_output_op_name_in_user_job() const { UNIMPLEMENTED(); }
  virtual void PushBlob(uint64_t ofblob_ptr) const { UNIMPLEMENTED(); }
  virtual void PullBlob(uint64_t ofblob_ptr) const { UNIMPLEMENTED(); }
  virtual void Finish() const { UNIMPLEMENTED(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
