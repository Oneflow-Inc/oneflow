#ifndef ONEFLOW_JOB_JOB_DESC_H_
#define ONEFLOW_JOB_JOB_DESC_H_

#include "common/util.h"
#include "job/job_conf.pb.h"

namespace oneflow {

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  ~JobDesc() = default;

  static JobDesc& Singleton() {
    static JobDesc obj;
    return obj;
  }

  void Init(const JobUserConf&) { TODO(); }
  
  const DLNetConf& train_dlnet_conf() const { TODO(); }
  const Resource& resource() const { TODO(); }
  const Strategy& strategy() const { TODO(); }

  const std::string& MdLoadMachine() { TODO(); }
  const std::string& MdSaveMachine() { TODO(); }

 private:
  JobDesc() = default;
};

} // namespace oneflow

#endif // ONEFLOW_JOB_JOB_DESC_H_
