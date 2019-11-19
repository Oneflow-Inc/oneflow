#ifndef ONEFLOW_CORE_JOB_CLUSTER_DESC_H_
#define ONEFLOW_CORE_JOB_CLUSTER_DESC_H_

#include "oneflow/core/job/cluster.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class ClusterDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClusterDesc);
  ~ClusterDesc() = default;

  size_t TotalMachineNum() const { return cluster_proto_.machine().size(); }
  const Machine& machine(int32_t idx) const { return cluster_proto_.machine(idx); }
  int32_t ctrl_port() const { return cluster_proto_.ctrl_port(); }
  int32_t data_port() const { return cluster_proto_.data_port(); }
  bool grpc_use_no_signal() const { return cluster_proto_.grpc_use_no_signal(); }
  int64_t GetMachineId(const std::string& addr) const;

 private:
  friend class Global<ClusterDesc>;
  explicit ClusterDesc(const ClusterProto& cluster_proto) : cluster_proto_(cluster_proto) {}

  ClusterProto cluster_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CLUSTER_DESC_H_
