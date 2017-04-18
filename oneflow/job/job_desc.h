#ifndef ONEFLOW_JOB_JOB_DESC_H_
#define ONEFLOW_JOB_JOB_DESC_H_

#include "common/util.h"

namespace oneflow {

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  ~JobDesc() = default;

  static JobDesc& Singleton() {
    static JobDesc obj;
    return obj;
  }

  const std::string& MdLoadMachine() { TODO(); }
  const std::string& MdSaveMachine() { TODO(); }

 private:
  JobDesc() = default;
};

} // namespace oneflow

#endif // ONEFLOW_JOB_JOB_DESC_H_
