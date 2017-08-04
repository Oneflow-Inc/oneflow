#ifndef ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_H_
#define ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/data_structure/node.h"
#include "oneflow/core/schedule/data_structure/schedule_result.h"
#include "oneflow/core/schedule/data_structure/session.h"
#include "oneflow/core/schedule/util/util.h"

namespace oneflow {
namespace schedule {

class PolicyHub;

class Policy {
 public:
  Policy() = default;
  explicit Policy(PolicyHub* ph) : ph_(ph) {}
  virtual ~Policy() = default;
  OF_DISALLOW_COPY_AND_MOVE(Policy);
  DEFINE_PURE_VIRTUAL_TYPE();
  inline const PolicyHub* ph() const { return ph_; }

 protected:
  PolicyHub* ph_;
};

class PrinterPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(PrinterPolicy);

  virtual void PrintGraph(const GraphNode& graph,
                          const std::string& filename) = 0;
};

class TestGraphGeneratorPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(TestGraphGeneratorPolicy);

  virtual std::unique_ptr<GraphNode> DemoGraph() = 0;
};

class GraphBuilderPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(GraphBuilderPolicy);

  virtual std::unique_ptr<GraphNode> BuildeGraph(const Plan& plan) = 0;
};

class StaticSchedulerPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(StaticSchedulerPolicy);

  virtual std::unique_ptr<Session> MakeSession(const GraphNode& graph) = 0;
  virtual std::unique_ptr<ScheduleResult> Schedule(const Session& session) = 0;
};

class ScheduleValidatorPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(ScheduleValidatorPolicy);
  virtual bool ValidateSchedule(const Session& session,
                                const ScheduleResult& result) = 0;
};

class RetimingPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(RetimingPolicy);
  virtual void Retiming(const Session& session, ScheduleResult* result) = 0;
};

class AllocatorPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(AllocatorPolicy);
  virtual void AllocateFromSchedule(const Session& session,
                                    ScheduleResult* result) = 0;
};

class AllocationValidatorPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(AllocationValidatorPolicy);
  virtual bool ValidateAllocation(const Session& session,
                                  const ScheduleResult& result) = 0;
};

class LimitedAllocatorPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(LimitedAllocatorPolicy);
  virtual std::unique_ptr<ScheduleResult> LimitedAllocate(
      const Session& session) = 0;
};

class LimitedAllocationValidatorPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(LimitedAllocationValidatorPolicy);
  virtual bool ValidateLimitedAllocation(const Session& session,
                                         const ScheduleResult& result) = 0;
};

class PlanSetterPolicy : public Policy {
 public:
  POLICY_INTERFACE_BOILERPLATE(PlanSetterPolicy);
  virtual bool SetPlanRegstNum(const ScheduleResult& result, Plan* plan) = 0;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_H_
