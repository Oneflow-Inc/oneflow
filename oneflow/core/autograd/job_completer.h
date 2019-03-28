#ifndef ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_
#define ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class JobCompleter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobCompleter);
  JobCompleter() = default;
  ~JobCompleter() = default;

  void Complete(JobConf1* job_conf_) const;
  
 private:
};

}

#endif  // ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_
