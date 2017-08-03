#ifndef ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_HUB_H_
#define ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_HUB_H_

#include "oneflow/core/schedule/interface/policy.h"
#include "oneflow/core/schedule/util/util.h"

namespace oneflow {
namespace schedule {

#define CLONE_POLICY(obj, getter)                                     \
  do {                                                                \
    if (obj.getter.get()) { mut_##getter = obj.getter->Clone(this); } \
  } while (0)

class PolicyHubBase {
 public:
  PolicyHubBase() = default;
  PolicyHubBase(const std::string& name) : name_(name) {}
  virtual ~PolicyHubBase() = default;
  OF_DISALLOW_COPY_AND_MOVE(PolicyHubBase);
  DEFINE_PURE_VIRTUAL_TYPE();

 private:
  std::string name_;
};

class PolicyHub : public PolicyHubBase {
 public:
  PolicyHub() = default;
  PolicyHub(const std::string& name) : PolicyHubBase(name) {}
  virtual ~PolicyHub() = default;
  OF_DISALLOW_COPY_AND_MOVE(PolicyHub);
  DEFINE_METHOD_TYPE();

  PolicyHub* Merge(const PolicyHub& ph) {
    CLONE_POLICY(ph, graph_builder());
    CLONE_POLICY(ph, limited_allocator());
    CLONE_POLICY(ph, printer());
    CLONE_POLICY(ph, test_graph_generator());
    CLONE_POLICY(ph, static_scheduler());
    CLONE_POLICY(ph, schedule_validator());
    CLONE_POLICY(ph, retiming());
    CLONE_POLICY(ph, allocator());
    CLONE_POLICY(ph, allocation_validator());
    CLONE_POLICY(ph, limited_allocation_validator());
    CLONE_POLICY(ph, plan_setter());
    return this;
  }

  PolicyHub* Merge(const PolicyHub* ph) { return Merge(*ph); }

  inline const std::unique_ptr<StaticSchedulerPolicy>& static_scheduler()
      const {
    return static_scheduler_;
  }
  inline std::unique_ptr<StaticSchedulerPolicy>& mut_static_scheduler() {
    return static_scheduler_;
  }
  PolicyHub* Add(std::unique_ptr<StaticSchedulerPolicy>&& policy) {
    mut_static_scheduler() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<ScheduleValidatorPolicy>& schedule_validator()
      const {
    return schedule_validator_;
  }
  inline std::unique_ptr<ScheduleValidatorPolicy>& mut_schedule_validator() {
    return schedule_validator_;
  }
  PolicyHub* Add(std::unique_ptr<ScheduleValidatorPolicy>&& policy) {
    mut_schedule_validator() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<RetimingPolicy>& retiming() const {
    return retiming_;
  }
  inline std::unique_ptr<RetimingPolicy>& mut_retiming() { return retiming_; }
  PolicyHub* Add(std::unique_ptr<RetimingPolicy>&& policy) {
    mut_retiming() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<AllocatorPolicy>& allocator() const {
    return allocator_;
  }
  inline std::unique_ptr<AllocatorPolicy>& mut_allocator() {
    return allocator_;
  }
  PolicyHub* Add(std::unique_ptr<AllocatorPolicy>&& policy) {
    mut_allocator() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<LimitedAllocationValidatorPolicy>&
  limited_allocation_validator() const {
    return limited_allocation_validator_;
  }
  inline std::unique_ptr<LimitedAllocationValidatorPolicy>&
  mut_limited_allocation_validator() {
    return limited_allocation_validator_;
  }
  PolicyHub* Add(std::unique_ptr<LimitedAllocationValidatorPolicy>&& policy) {
    mut_limited_allocation_validator() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<GraphBuilderPolicy>& graph_builder() const {
    return graph_builder_;
  }
  inline std::unique_ptr<GraphBuilderPolicy>& mut_graph_builder() {
    return graph_builder_;
  }
  PolicyHub* Add(std::unique_ptr<GraphBuilderPolicy>&& policy) {
    mut_graph_builder() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<LimitedAllocatorPolicy>& limited_allocator()
      const {
    return limited_allocator_;
  }
  inline std::unique_ptr<LimitedAllocatorPolicy>& mut_limited_allocator() {
    return limited_allocator_;
  }
  PolicyHub* Add(std::unique_ptr<LimitedAllocatorPolicy>&& policy) {
    mut_limited_allocator() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<PrinterPolicy>& printer() const {
    return printer_;
  }
  inline std::unique_ptr<PrinterPolicy>& mut_printer() { return printer_; }
  PolicyHub* Add(std::unique_ptr<PrinterPolicy>&& policy) {
    mut_printer() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<TestGraphGeneratorPolicy>& test_graph_generator()
      const {
    return test_graph_generator_;
  }
  inline std::unique_ptr<TestGraphGeneratorPolicy>& mut_test_graph_generator() {
    return test_graph_generator_;
  }
  PolicyHub* Add(std::unique_ptr<TestGraphGeneratorPolicy>&& policy) {
    mut_test_graph_generator() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<AllocationValidatorPolicy>&
  allocation_validator() const {
    return allocation_validator_;
  }
  inline std::unique_ptr<AllocationValidatorPolicy>&
  mut_allocation_validator() {
    return allocation_validator_;
  }
  PolicyHub* Add(std::unique_ptr<AllocationValidatorPolicy>&& policy) {
    mut_allocation_validator() = std::move(policy);
    return this;
  }

  inline const std::unique_ptr<PlanSetterPolicy>& plan_setter() const {
    return plan_setter_;
  }
  inline std::unique_ptr<PlanSetterPolicy>& mut_plan_setter() {
    return plan_setter_;
  }
  PolicyHub* Add(std::unique_ptr<PlanSetterPolicy>&& policy) {
    mut_plan_setter() = std::move(policy);
    return this;
  }

 private:
  std::unique_ptr<GraphBuilderPolicy> graph_builder_;
  std::unique_ptr<LimitedAllocatorPolicy> limited_allocator_;
  std::unique_ptr<PrinterPolicy> printer_;
  std::unique_ptr<TestGraphGeneratorPolicy> test_graph_generator_;
  std::unique_ptr<StaticSchedulerPolicy> static_scheduler_;
  std::unique_ptr<ScheduleValidatorPolicy> schedule_validator_;
  std::unique_ptr<RetimingPolicy> retiming_;
  std::unique_ptr<AllocatorPolicy> allocator_;
  std::unique_ptr<AllocationValidatorPolicy> allocation_validator_;
  std::unique_ptr<LimitedAllocationValidatorPolicy>
      limited_allocation_validator_;
  std::unique_ptr<PlanSetterPolicy> plan_setter_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_INTERFACE_POLICY_HUB_H_
