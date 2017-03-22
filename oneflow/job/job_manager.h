#ifndef ONEFLOW_JOB_JOB_MANAGER_H_
#define ONEFLOW_JOB_JOB_MANAGER_H_

#include "common/util.h"
#include "job/job_conf.pb.h"

namespace oneflow {

class JobManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobManager);

  JobManager() = default;
  ~JobManager() = default;

  void Init(const JobUserConf& job_user_conf); // TODO: implement it

  void compile(); // TODO: implement it
  void run(); // TODO: implement it

 private:
  JobSysConf job_sys_conf_;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_JOB_MANAGER_H_
