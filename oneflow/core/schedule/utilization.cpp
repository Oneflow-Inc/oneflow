#include "oneflow/core/schedule/utilization.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

std::string Utilization::VisualStr() const {
  std::stringstream ss;
  ss << UtilizationUtil::GetUniqueName(utilization_proto().resource(), "\\n")
     << "\\nTime: " << std::setprecision(3) << utilization_proto().start_at()
     << "-" << std::setprecision(3) << utilization_proto().end_at()
     << "\\nUtilization: " << std::setprecision(3)
     << utilization_proto().utilization()
     << "\\nBatch: " << utilization_proto().start_batch_id() << "-"
     << utilization_proto().end_batch_id()
     << "\\nRecordCount: " << raw_protos().size();
  return ss.str();
}

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
  float utilization = total_time / (parallel_num * (end_at - start_at));
  CHECK(utilization >= 0);
  CHECK(utilization <= 1);
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

float Utilization::GetTimePerBatch(const UtilizationGraph& ugraph) const {
  float total_time =
      utilization_proto().utilization() * ParallelNum(ugraph)
      * (utilization_proto().end_at() - utilization_proto().start_at());
  int32_t batch_num = utilization_proto().end_batch_id()
                      - utilization_proto().start_batch_id() + 1;
  CHECK(batch_num > 0);
  CHECK(batch_num <= utilization_proto().end_batch_id());
  return total_time / batch_num;
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

uint32_t TaskUtilization::ParallelNum(const UtilizationGraph& ugraph) const {
  std::list<TaskStreamUtilization*> l;
  ugraph.arc_mgr<TaskUtilization, TaskStreamUtilization>().Output(
      const_cast<TaskUtilization*>(this), &l);
  return ugraph.arc_mgr<StreamUtilization, TaskStreamUtilization>().Input(l);
}

}  // namespace schedule
}  // namespace oneflow
