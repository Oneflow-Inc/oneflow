#ifndef ONEFLOW_CORE_JOB_IMPROVER_H_
#define ONEFLOW_CORE_JOB_IMPROVER_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/graph/act_graph.h"

namespace oneflow {

class Improver final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Improver);
  Improver() = delete;
  ~Improver() = default;

  OF_SINGLETON(Improver);

  Plan Improve(const Plan& naive_plan, const std::string& act_event_filepath);

 private:
  explicit Improver(const AvailableMemDesc& amd) : amd_(amd) {}
  void MemoryLimitedAllocate(const ActGraph& graph, double base_ii,
                             HashMap<int64_t, double>* regst_desc_id2num) const;
  using MemZoneRegstDescs =
      std::vector<std::vector<std::list<const RegstDescProto*>>>;
  bool IsAnyZoneOutOfMemory(
      const MemZoneRegstDescs& mz_regst_descs,
      const HashMap<int64_t, double>& regst_desc_id2life_time, double ii) const;
  double CalcII(double ii_base,
                HashMap<int64_t, double> regst_desc_id2life_time,
                const MemZoneRegstDescs& mz_regst_descs) const;
  size_t AvailableMemSize(int64_t machine_id, int64_t memory_zone_id) const;
  int64_t GetMemoryZoneId(const MemoryCase& mem_case) const;
  void MakeMemZoneRegstDescs(const Plan& plan,
                             MemZoneRegstDescs* mz2regst_desc) const;
  AvailableMemDesc amd_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IMPROVER_H_
