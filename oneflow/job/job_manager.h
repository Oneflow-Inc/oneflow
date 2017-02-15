#ifndef ONEFLOW_JOB_JOB_MANAGER_H
#define ONEFLOW_JOB_JOB_MANAGER_H

#include "job/job_conf.pb.h"

namespace oneflow {

class JobManager {
 public:
  JobManager() = default;
  JobManager(const JobManager&) = delete;
  JobManager(JobManager&&) = delete;
  JobManager& operator = (const JobManager&) = delete;
  JobManager& operator = (JobManager&&) = delete;
  ~JobManager() = default;

  void init(const JobUserConf& job_user_conf); // TODO: implement it

  void compile(); // TODO: implement it
  void run(); // TODO: implement it

 private:
  JobSysConf job_sys_conf_;

};

} // namespace oneflow

#endif
