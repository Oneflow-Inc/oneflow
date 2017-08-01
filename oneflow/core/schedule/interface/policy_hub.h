#ifndef ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_HUB_H_
#define ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_HUB_H_

#include "oneflow/core/schedule/interface/policy.h"
#include "oneflow/core/schedule/utils/utils.h"

namespace oneflow {
namespace schedule {

class PolicyHubBase {
 public:
  PolicyHubBase() = default;
  virtual ~PolicyHubBase() = default;
  OF_DISALLOW_COPY_AND_MOVE(PolicyHubBase);
  DEFINE_PURE_VIRTUAL_TYPE();

  inline const GraphBuilderPolicy* graph_builder() const {
    return graph_builder_.get();
  }
  inline const LimitedAllocatorPolicy* limited_allocator() const {
    return limited_allocator_.get();
  }

  inline std::shared_ptr<GraphBuilderPolicy>& mut_graph_builder() {
    return graph_builder_;
  }
  inline std::shared_ptr<LimitedAllocatorPolicy>& mut_limited_allocator() {
    return limited_allocator_;
  }

 private:
  std::shared_ptr<GraphBuilderPolicy> graph_builder_;
  std::shared_ptr<LimitedAllocatorPolicy> limited_allocator_;
};

class PolicyHub : public PolicyHubBase {
 public:
  PolicyHub() = default;
  virtual ~PolicyHub() = default;
  OF_DISALLOW_COPY_AND_MOVE(PolicyHub);
  DEFINE_METHOD_TYPE();

  inline const GraphPrinterPolicy* graph_printer() const {
    return graph_printer_.get();
  }
  inline const TestGraphGeneratorPolicy* test_graph_generator() const {
    return test_graph_generator_.get();
  }
  inline const StaticSchedulerPolicy* static_scheduler() const {
    return static_scheduler_.get();
  }
  inline const ScheduleValidatorPolicy* schedule_validator() const {
    return schedule_validator_.get();
  }
  inline const RetimingPolicy* retiming() const { return retiming_.get(); }
  inline const AllocatorPolicy* allocator() const { return allocator_.get(); }
  inline const AllocationValidatorPolicy* allocation_validator() const {
    return allocation_validator_.get();
  }
  inline const LimitedAllocationValidatorPolicy* limited_allocation_validator()
      const {
    return limited_allocation_validator_.get();
  }

  inline std::shared_ptr<GraphPrinterPolicy>& mut_graph_printer() {
    return graph_printer_;
  }
  inline std::shared_ptr<TestGraphGeneratorPolicy>& mut_test_graph_generator() {
    return test_graph_generator_;
  }
  inline std::shared_ptr<StaticSchedulerPolicy>& mut_static_scheduler() {
    return static_scheduler_;
  }
  inline std::shared_ptr<ScheduleValidatorPolicy>& mut_schedule_validator() {
    return schedule_validator_;
  }
  inline std::shared_ptr<RetimingPolicy>& mut_retiming() { return retiming_; }
  inline std::shared_ptr<AllocatorPolicy>& mut_allocator() {
    return allocator_;
  }
  inline std::shared_ptr<AllocationValidatorPolicy>&
  mut_allocation_validator() {
    return allocation_validator_;
  }
  inline std::shared_ptr<LimitedAllocationValidatorPolicy>&
  mut_limited_allocation_validator() {
    return limited_allocation_validator_;
  }

 private:
  std::shared_ptr<GraphPrinterPolicy> graph_printer_;
  std::shared_ptr<TestGraphGeneratorPolicy> test_graph_generator_;
  std::shared_ptr<StaticSchedulerPolicy> static_scheduler_;
  std::shared_ptr<ScheduleValidatorPolicy> schedule_validator_;
  std::shared_ptr<RetimingPolicy> retiming_;
  std::shared_ptr<AllocatorPolicy> allocator_;
  std::shared_ptr<AllocationValidatorPolicy> allocation_validator_;
  std::shared_ptr<LimitedAllocationValidatorPolicy>
      limited_allocation_validator_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_HUB_H_
