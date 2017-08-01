#ifndef ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_H_
#define ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_H_

#include "oneflow/core/schedule/data_structure/node.h"
#include "oneflow/core/schedule/utils/utils.h"

namespace oneflow {
namespace schedule {

class PolicyHub;

class Policy {
 public:
  Policy() = default;
  explicit Policy(PolicyHub* pb) : pb_(pb) {}
  virtual ~Policy() = default;
  OF_DISALLOW_COPY_AND_MOVE(Policy);
  DEFINE_PURE_VIRTUAL_TYPE();
  inline const PolicyHub* pb() const { return pb_; }

 protected:
  PolicyHub* pb_;
};

class GraphPrinterPolicy : public Policy {
 public:
  GraphPrinterPolicy() = default;
  explicit GraphPrinterPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~GraphPrinterPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(GraphPrinterPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();

  virtual void ConsolePrint(const GraphNode& graph) = 0;
};

class TestGraphGeneratorPolicy : public Policy {
 public:
  TestGraphGeneratorPolicy() = default;
  explicit TestGraphGeneratorPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~TestGraphGeneratorPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(TestGraphGeneratorPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();

  virtual std::unique_ptr<GraphNode> Demo() = 0;
};

class GraphBuilderPolicy : public Policy {
 public:
  GraphBuilderPolicy() = default;
  explicit GraphBuilderPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~GraphBuilderPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(GraphBuilderPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();
};

class StaticSchedulerPolicy : public Policy {
 public:
  StaticSchedulerPolicy() = default;
  explicit StaticSchedulerPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~StaticSchedulerPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(StaticSchedulerPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();
};

class ScheduleValidatorPolicy : public Policy {
 public:
  ScheduleValidatorPolicy() = default;
  explicit ScheduleValidatorPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~ScheduleValidatorPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(ScheduleValidatorPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();
};

class RetimingPolicy : public Policy {
 public:
  RetimingPolicy() = default;
  explicit RetimingPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~RetimingPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(RetimingPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();
};

class AllocatorPolicy : public Policy {
 public:
  AllocatorPolicy() = default;
  explicit AllocatorPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~AllocatorPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(AllocatorPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();
};

class AllocationValidatorPolicy : public Policy {
 public:
  AllocationValidatorPolicy() = default;
  explicit AllocationValidatorPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~AllocationValidatorPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(AllocationValidatorPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();
};

class LimitedAllocatorPolicy : public Policy {
 public:
  LimitedAllocatorPolicy() = default;
  explicit LimitedAllocatorPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~LimitedAllocatorPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(LimitedAllocatorPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();
};

class LimitedAllocationValidatorPolicy : public Policy {
 public:
  LimitedAllocationValidatorPolicy() = default;
  explicit LimitedAllocationValidatorPolicy(PolicyHub* pb) : Policy(pb) {}
  virtual ~LimitedAllocationValidatorPolicy() = default;
  OF_DISALLOW_COPY_AND_MOVE(LimitedAllocationValidatorPolicy);
  DEFINE_PURE_VIRTUAL_TYPE();
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_H_
