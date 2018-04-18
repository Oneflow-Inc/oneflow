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

  explicit Improver(const AvailableMemDesc& amd) : amd_(amd) {}

  Plan Improve(const Plan& naive_plan, const std::string& act_event_filepath);

 private:
  void MemoryLimitedAllocate(
      const ActGraph& graph,
      const std::function<void(int64_t, uint64_t)>& Handler) const;

  //  first dimension index of MemZoneRegstDescs is machine_id
  //  second dimension index of MemZoneRegstDescs is mem_zone_id
  using MemZoneRegstDescs =
      std::vector<std::vector<std::list<const RegstDescProto*>>>;
  bool IsAnyZoneOutOfMemory(
      const MemZoneRegstDescs& mz_regst_descs,
      const std::function<const HashMap<int64_t, double>&(int64_t)>&
          Duration4RegstDescId,
      const std::function<const HashMap<int64_t, double>&(int64_t)>&
          Ratio4RegstDescId,
      double ii) const;
  double BinarySearchII(
      double base_ii,
      const std::function<const HashMap<int64_t, double>&(int64_t)>&
          Duration4RegstDescId,
      const std::function<const HashMap<int64_t, double>&(int64_t)>&
          Ratio4RegstDescId,
      const MemZoneRegstDescs& mz_regst_descs) const;
  uint64_t AvailableMemSize(int64_t machine_id, int64_t memory_zone_id) const;
  int64_t GetMemoryZoneId(const MemoryCase& mem_case) const;
  void MakeMemZoneRegstDescs(const Plan& plan,
                             MemZoneRegstDescs* mz2regst_desc) const;
  double CalcMaxRegstDescDuration(
      const std::function<const HashMap<int64_t, double>&(int64_t)>&
          Duration4RegstDescId,

      const MemZoneRegstDescs& mz_regst_descs) const;

  AvailableMemDesc amd_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IMPROVER_H_
