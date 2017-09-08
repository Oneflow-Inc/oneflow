#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/snode.h"
#include "oneflow/core/schedule/utilization.pb.h"

namespace oneflow {
namespace schedule {

class UtilizationGraph;

class Utilization : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Utilization);
  explicit Utilization(const std::string name) : SNode(name) {}
  virtual ~Utilization() = default;

  inline UtilizationProto* mut_utilization_proto() {
    return &utilization_proto_;
  }
  inline const UtilizationProto& utilization_proto() const {
    return utilization_proto_;
  }

  virtual void CreateAscendantIfNotFound(UtilizationGraph* graph) const = 0;
  virtual bool IsLeaf() const { return false; }

 private:
  UtilizationProto utilization_proto_;
};

class ComputationUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ComputationUtilization);
  ComputationUtilization() : Utilization("") {
    mut_utilization_proto()->mutable_dev_computation_resource()->set_device_id(
        0u);
  }
  ~ComputationUtilization() = default;

  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override {}
};

class DeviceComputationUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceComputationUtilization);
  explicit DeviceComputationUtilization(uint64_t device_id = 0u)
      : Utilization(MakeUniqueName(device_id)) {
    mut_utilization_proto()->mutable_dev_computation_resource()->set_device_id(
        device_id);
  }
  ~DeviceComputationUtilization() = default;

  inline uint64_t device_id() const {
    return utilization_proto().dev_computation_resource().device_id();
  }
  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override;
  static std::string MakeUniqueName(uint64_t device_id) {
    return std::to_string(device_id);
  }
};

class StreamUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StreamUtilization);
  StreamUtilization(uint64_t device_id, uint64_t stream_id)
      : Utilization(MakeUniqueName(device_id, stream_id)) {
    auto stream = mut_utilization_proto()->mutable_stream_resource();
    stream->set_device_id(device_id);
    stream->set_stream_id(stream_id);
  }
  ~StreamUtilization() = default;

  inline uint64_t device_id() const {
    return utilization_proto().stream_resource().device_id();
  }
  inline uint64_t stream_id() const {
    return utilization_proto().stream_resource().stream_id();
  }
  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override;
  static std::string MakeUniqueName(uint64_t device_id, uint64_t stream_id) {
    return std::to_string(device_id) + "," + std::to_string(stream_id);
  }
};

class TaskUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskUtilization);
  TaskUtilization(uint64_t task_id) : Utilization(MakeUniqueName(task_id)) {
    auto task = mut_utilization_proto()->mutable_task_resource();
    task->set_task_id(task_id);
  }
  ~TaskUtilization() = default;

  inline uint64_t task_id() const {
    return utilization_proto().task_resource().task_id();
  }
  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override;
  static std::string MakeUniqueName(uint64_t task_id) {
    return std::to_string(task_id);
  }
};

class TaskStreamUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskStreamUtilization);
  TaskStreamUtilization(uint64_t task_id, uint64_t stream_id)
      : Utilization(MakeUniqueName(task_id, stream_id)) {
    auto task_stream = mut_utilization_proto()->mutable_task_stream_resource();
    task_stream->set_task_id(task_id);
    task_stream->set_stream_id(stream_id);
  }
  ~TaskStreamUtilization() = default;

  inline uint64_t task_id() const {
    return utilization_proto().task_stream_resource().task_id();
  }
  inline uint64_t stream_id() const {
    return utilization_proto().task_stream_resource().stream_id();
  }
  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override;
  virtual bool IsLeaf() const override { return false; }
  static std::string MakeUniqueName(uint64_t task_id, uint64_t stream_id) {
    return std::to_string(task_id) + "," + std::to_string(stream_id);
  }
};

class MemoryUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryUtilization);
  MemoryUtilization() : Utilization("") {
    mut_utilization_proto()->mutable_dev_memory_resource()->set_device_id(0u);
  }
  ~MemoryUtilization() = default;

  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override {}
};

class DeviceMemoryUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceMemoryUtilization);
  explicit DeviceMemoryUtilization(uint64_t device_id)
      : Utilization(MakeUniqueName(device_id)) {
    mut_utilization_proto()->mutable_dev_memory_resource()->set_device_id(
        device_id);
  }
  ~DeviceMemoryUtilization() = default;

  inline uint64_t device_id() const {
    return utilization_proto().dev_memory_resource().device_id();
  }

  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override;
  static std::string MakeUniqueName(uint64_t device_id) {
    return std::to_string(device_id);
  }
};

class RegstDescUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDescUtilization);
  explicit RegstDescUtilization(uint64_t regst_desc_id)
      : Utilization(MakeUniqueName(regst_desc_id)) {
    mut_utilization_proto()->mutable_regst_desc_resource()->set_regst_desc_id(
        regst_desc_id);
  }
  ~RegstDescUtilization() = default;

  inline uint64_t regst_desc_id() const {
    return utilization_proto().regst_desc_resource().regst_desc_id();
  }

  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override;
  static std::string MakeUniqueName(uint64_t regst_desc_id) {
    return std::to_string(regst_desc_id);
  }
};

class RegstUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstUtilization);
  RegstUtilization(uint64_t regst_desc_id, uint64_t regst_id)
      : Utilization(MakeUniqueName(regst_desc_id, regst_id)) {
    auto regst = mut_utilization_proto()->mutable_regst_resource();
    regst->set_regst_desc_id(regst_desc_id);
    regst->set_regst_id(regst_id);
  }
  ~RegstUtilization() = default;

  inline uint64_t regst_desc_id() const {
    return utilization_proto().regst_resource().regst_desc_id();
  }
  inline uint64_t regst_id() const {
    return utilization_proto().regst_resource().regst_id();
  }
  void CreateAscendantIfNotFound(UtilizationGraph* graph) const override;
  virtual bool IsLeaf() const override { return false; }
  static std::string MakeUniqueName(uint64_t regst_desc_id, uint64_t regst_id) {
    return std::to_string(regst_desc_id) + "," + std::to_string(regst_id);
  }
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_
