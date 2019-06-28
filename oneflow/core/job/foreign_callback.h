#ifndef ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H
#define ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H

namespace oneflow {

class ForeignCallback {
 public:
  ForeignCallback() = default;

  virtual ~ForeignCallback() = default;

  virtual void run() = 0;
};

}  // namespace oneflow

#endif
