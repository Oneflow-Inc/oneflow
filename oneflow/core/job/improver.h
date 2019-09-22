#ifndef ONEFLOW_CORE_JOB_IMPROVER_H_
#define ONEFLOW_CORE_JOB_IMPROVER_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/graph/chain_act_graph.h"

namespace oneflow {

class Improver final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Improver);
  Improver() : start_mem_block_id_(-1) {}
  ~Improver() = default;

  Plan Improve(const AvailableMemDesc& amd, const Plan& naive_plan,
               const std::string& act_event_filepath);
  Plan GenAndInferMemBlockIdOnly(const AvailableMemDesc& amd, const Plan& naive_plan);

 private:
  Plan GenAndInferMemBlockId(const Plan& naive_plan) const;
  void Init(const AvailableMemDesc& amd, const Plan& naive_plan);
  void ForEachImprovedRegstNum(
      const Plan& plan, bool is_memory_limited, double ii,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
      const std::function<void(int64_t, uint64_t)>& Handler) const;
  void ForEachInferredMemSharingCriticalSection(
      const Plan& plan, const std::function<int64_t(int64_t)>& OrderInGraph4TaskId,
      const std::function<void(const std::vector<const RegstDescProto*>&)>& Handler) const;
  //  first dimension index of MemZoneRegstDescs is machine_id
  //  second dimension index of MemZoneRegstDescs is mem_zone_id
  using MemZoneRegstDescs = std::vector<std::vector<std::list<const RegstDescProto*>>>;
  bool IsAnyZoneOutOfMemory(
      const MemZoneRegstDescs& mz_regst_descs,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Duration4RegstDescId,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Ratio4RegstDescId,
      double ii) const;
  double BinarySearchII(
      double base_ii,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Duration4RegstDescId,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Ratio4RegstDescId,
      const MemZoneRegstDescs& mz_regst_descs) const;
  uint64_t AvailableMemSize(int64_t machine_id, int64_t memory_zone_id) const;
  int64_t GetMemoryZoneId(const MemoryCase& mem_case) const;
  void MakeMemZoneRegstDescs(const Plan& plan, MemZoneRegstDescs* mz2regst_desc) const;
  double CalcMaxRegstDescDuration(
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Duration4RegstDescId,
      const MemZoneRegstDescs& mz_regst_descs) const;

  int32_t start_mem_block_id_;
  AvailableMemDesc amd_;
  std::vector<int32_t> record_load_task_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IMPROVER_H_
