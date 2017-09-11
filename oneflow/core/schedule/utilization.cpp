#include "oneflow/core/schedule/utilization.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

void Utilization::Reduce(const UtilizationGraph& graph) {
  uint32_t parallel_num = ParallelNum(graph);
  CHECK(parallel_num);
  float total_time = 0, start_at = UINT64_MAX, end_at = 0;
  uint64_t start_batch_id = UINT64_MAX, end_batch_id = 0;
  for (const UtilizationProto* u : raw_protos_) {
    total_time += u->end_at() - u->start_at();
    start_at = std::min(start_at, u->start_at());
    end_at = std::max(end_at, u->end_at());
    start_batch_id = std::min(start_batch_id, u->start_batch_id());
    end_batch_id = std::max(end_batch_id, u->end_batch_id());
  }
  CHECK(end_at > start_at);
  float utilization = total_time / parallel_num * (end_at - start_at);
  CHECK(total_time < 1);
  utilization_proto_.set_utilization(utilization);
  utilization_proto_.set_start_at(start_at);
  utilization_proto_.set_start_at(start_at);
  utilization_proto_.set_end_at(end_at);
  utilization_proto_.set_start_batch_id(start_batch_id);
  utilization_proto_.set_end_batch_id(end_batch_id);
}

uint32_t MemoryUtilization::ParallelNum(const UtilizationGraph& graph) const {
  return graph.inner2leaf_arc_mgr().Output(
      const_cast<MemoryUtilization*>(this));
}

uint32_t ComputationUtilization::ParallelNum(
    const UtilizationGraph& graph) const {
  uint64_t stream_cnt = 0;
  graph.arc_mgr<ComputationUtilization, DeviceComputationUtilization>().Output(
      const_cast<ComputationUtilization*>(this),
      [&](DeviceComputationUtilization* dev_c_utilization) {
        stream_cnt += dev_c_utilization->ParallelNum(graph);
      });
  return stream_cnt;
}

uint32_t DeviceComputationUtilization::ParallelNum(
    const UtilizationGraph& graph) const {
  return graph.arc_mgr<DeviceComputationUtilization, StreamUtilization>()
      .Output(const_cast<DeviceComputationUtilization*>(this));
}

void Utilization::CreateAscendantIfNotFound(UtilizationGraph* ugraph) const {
  auto nonconst_this = const_cast<Utilization*>(this);
  UtilizationUtil::ForEachGrouped(
      utilization_proto().resource(), *ugraph,
      [&](const UtilizationResource& grouped_resource) {
        Utilization* grouped =
            ugraph->FindOrCreateUtilization(grouped_resource);
        ugraph->Connect(grouped, nonconst_this);
      });
}

}  // namespace schedule
}  // namespace oneflow
