#ifndef ONEFLOW_CORE_JOB_CLUSTER_DESC_H_
#define ONEFLOW_CORE_JOB_CLUSTER_DESC_H_

#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class EnvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EnvDesc);
  ~EnvDesc() = default;

  size_t TotalMachineNum() const { return env_proto_.machine().size(); }
  const Machine& machine(int32_t idx) const { return env_proto_.machine(idx); }
  int32_t ctrl_port() const { return env_proto_.ctrl_port(); }
  int32_t data_port() const { return env_proto_.data_port(); }
  bool grpc_use_no_signal() const { return env_proto_.grpc_use_no_signal(); }
  int64_t GetMachineId(const std::string& addr) const;

 private:
  friend class Global<EnvDesc>;
  explicit EnvDesc(const EnvProto& env_proto) : env_proto_(env_proto) {}

  EnvProto env_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CLUSTER_DESC_H_
