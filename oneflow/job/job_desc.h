#ifndef ONEFLOW_JOB_JOB_DESC_H_
#define ONEFLOW_JOB_JOB_DESC_H_

#include "common/util.h"
#include "job/job_conf.pb.h"
#include "job/job_desc.pb.h"

namespace oneflow {

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  ~JobDesc() = default;

  static JobDesc& Singleton() {
    static JobDesc obj;
    return obj;
  }

  void InitFromJobConf(const JobConf&);
  void InitFromProto(const JobDescProto&);
  JobDescProto ToProto() const;
  
  // Getters
  const DLNetConf& train_dlnet_conf() const { return train_dl_net_conf_; }
  const Resource& resource() const { return resource_; }
  const Strategy& strategy() const { return strategy_; }

  const std::string& md_load_machine() { return md_load_machine_; }
  const std::string& md_save_machine() { return md_save_machine_; }

 private:
  JobDesc() = default;

  DLNetConf train_dl_net_conf_;
  Resource resource_;
  Strategy strategy_;
  std::string md_load_machine_;
  std::string md_save_machine_;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_JOB_DESC_H_
