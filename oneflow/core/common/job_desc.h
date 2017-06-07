#ifndef ONEFLOW_CORE_COMMON_JOB_DESC_H_
#define ONEFLOW_CORE_COMMON_JOB_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/conf/job_conf.pb.h"
#include "oneflow/core/common/job_desc.pb.h"

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
  void ToProto(JobDescProto*) const;
  
  // Getters
  const DLNetConf& train_dlnet_conf() const { return train_dlnet_conf_; }
  const Resource& resource() const { return resource_; }
  const Strategy& strategy() const { return strategy_; }
  const FloatingPointType& floating_point_type() const { 
    return floating_point_type_; 
  }
  const std::string& md_load_machine() { return md_load_machine_; }
  const std::string& md_save_machine() { return md_save_machine_; }
  bool is_train() const { return is_train_; }

  uint32_t piece_size() const { return piece_size_; }
  void set_piece_size(uint32_t val) { piece_size_ = val; }

 private:
  JobDesc() = default;

  DLNetConf train_dlnet_conf_;
  Resource resource_;
  Strategy strategy_;
  std::string md_load_machine_;
  std::string md_save_machine_;
  uint32_t batch_size_;
  uint32_t piece_size_;
  bool is_train_;
  FloatingPointType floating_point_type_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_COMMON_JOB_DESC_H_
