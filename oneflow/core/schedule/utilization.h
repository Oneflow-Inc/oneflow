#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/snode.h"
#include "oneflow/core/schedule/utilization.pb.h"

#define UTILIZATION_TYPE_SEQ                                                  \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kComputation,                     \
                       ComputationUtilization)                                \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kDevComputation,                  \
                       DeviceComputationUtilization)                          \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kStream, StreamUtilization)       \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kTask, TaskUtilization)           \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kTaskStream,                      \
                       TaskStreamUtilization)                                 \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kMemory, MemoryUtilization)       \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kDevMemory,                       \
                       DeviceMemoryUtilization)                               \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kRegstDesc, RegstDescUtilization) \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kRegst, RegstUtilization)

namespace oneflow {
namespace schedule {

class UtilizationGraph;

class Utilization : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Utilization);
  explicit Utilization(const std::string name) : SNode(name) {}
  virtual ~Utilization() = default;

  inline const UtilizationProto& utilization_proto() const {
    return utilization_proto_;
  }

  virtual uint32_t ParallelNum(const UtilizationGraph&) const { return 1u; }
  void Reduce(const UtilizationGraph&);
  virtual bool IsLeaf() const { return false; }

  inline std::list<const UtilizationProto*>* mut_raw_protos() {
    return &raw_protos_;
  }

 protected:
  inline UtilizationProto* mut_utilization_proto() {
    return &utilization_proto_;
  }

 private:
  UtilizationProto utilization_proto_;
  std::list<const UtilizationProto*> raw_protos_;
};

class TaskStreamUtilization;

class ComputationUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ComputationUtilization);
  ComputationUtilization(const std::string name = "") : Utilization(name) {
    mut_utilization_proto()->mutable_dev_computation_resource()->set_device_id(
        0u);
  }
  ~ComputationUtilization() = default;

  virtual void CreateAscendantIfNotFound(UtilizationGraph* graph,
                                         TaskStreamUtilization* leaf) const {}
};

class DeviceComputationUtilization : public ComputationUtilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceComputationUtilization);
  explicit DeviceComputationUtilization(uint64_t device_id = 0u)
      : ComputationUtilization(MakeUniqueName(device_id)) {
    mut_utilization_proto()->mutable_dev_computation_resource()->set_device_id(
        device_id);
  }
  ~DeviceComputationUtilization() = default;

  uint32_t ParallelNum(const UtilizationGraph& ugraph) const override;

  inline uint64_t device_id() const {
    return utilization_proto().dev_computation_resource().device_id();
  }
  void CreateAscendantIfNotFound(UtilizationGraph* graph,
                                 TaskStreamUtilization* leaf) const override;
  static std::string MakeUniqueName(uint64_t device_id) {
    return std::to_string(device_id);
  }
};

class StreamUtilization : public ComputationUtilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StreamUtilization);
  StreamUtilization(uint64_t device_id, uint64_t stream_id)
      : ComputationUtilization(MakeUniqueName(device_id, stream_id)) {
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
  void CreateAscendantIfNotFound(UtilizationGraph* graph,
                                 TaskStreamUtilization* leaf) const override;
  static std::string MakeUniqueName(uint64_t device_id, uint64_t stream_id) {
    return std::to_string(device_id) + "," + std::to_string(stream_id);
  }
};

class TaskUtilization : public ComputationUtilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskUtilization);
  TaskUtilization(uint64_t task_id)
      : ComputationUtilization(MakeUniqueName(task_id)) {
    auto task = mut_utilization_proto()->mutable_task_resource();
    task->set_task_id(task_id);
  }
  ~TaskUtilization() = default;

  inline uint64_t task_id() const {
    return utilization_proto().task_resource().task_id();
  }
  void CreateAscendantIfNotFound(UtilizationGraph* graph,
                                 TaskStreamUtilization* leaf) const override;
  static std::string MakeUniqueName(uint64_t task_id) {
    return std::to_string(task_id);
  }
};

class TaskStreamUtilization : public ComputationUtilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskStreamUtilization);
  TaskStreamUtilization(uint64_t task_id, uint64_t stream_id)
      : ComputationUtilization(MakeUniqueName(task_id, stream_id)) {
    InitKeyField(task_id, stream_id);
  }
  explicit TaskStreamUtilization(const UtilizationProto& proto)
      : ComputationUtilization(MakeUniqueName(proto)) {
    InitKeyField(proto.task_stream_resource().task_id(),
                 proto.task_stream_resource().stream_id());
  }
  ~TaskStreamUtilization() = default;

  void InitKeyField(uint64_t task_id, uint64_t stream_id) {
    auto task_stream = mut_utilization_proto()->mutable_task_stream_resource();
    task_stream->set_task_id(task_id);
    task_stream->set_stream_id(stream_id);
  }

  inline uint64_t task_id() const {
    return utilization_proto().task_stream_resource().task_id();
  }
  inline uint64_t stream_id() const {
    return utilization_proto().task_stream_resource().stream_id();
  }
  void CreateAscendantIfNotFound(
      UtilizationGraph* graph,
      TaskStreamUtilization* leaf = nullptr) const override;
  virtual bool IsLeaf() const override { return false; }
  static std::string MakeUniqueName(const UtilizationProto& proto) {
    return MakeUniqueName(proto.task_stream_resource().task_id(),
                          proto.task_stream_resource().stream_id());
  }
  static std::string MakeUniqueName(uint64_t task_id, uint64_t stream_id) {
    return std::to_string(task_id) + "," + std::to_string(stream_id);
  }
};

class RegstUtilization;
class MemoryUtilization : public Utilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryUtilization);
  MemoryUtilization(const std::string name = "") : Utilization(name) {
    mut_utilization_proto()->mutable_dev_memory_resource()->set_device_id(0u);
  }
  ~MemoryUtilization() = default;

  uint32_t ParallelNum(const UtilizationGraph& ugraph) const override;
  virtual void CreateAscendantIfNotFound(UtilizationGraph* graph,
                                         RegstUtilization* leaf) const {}
};

class DeviceMemoryUtilization : public MemoryUtilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceMemoryUtilization);
  explicit DeviceMemoryUtilization(uint64_t device_id)
      : MemoryUtilization(MakeUniqueName(device_id)) {
    mut_utilization_proto()->mutable_dev_memory_resource()->set_device_id(
        device_id);
  }
  ~DeviceMemoryUtilization() = default;

  inline uint64_t device_id() const {
    return utilization_proto().dev_memory_resource().device_id();
  }

  void CreateAscendantIfNotFound(UtilizationGraph* graph,
                                 RegstUtilization* leaf) const override;
  static std::string MakeUniqueName(uint64_t device_id) {
    return std::to_string(device_id);
  }
};

class RegstDescUtilization : public MemoryUtilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDescUtilization);
  explicit RegstDescUtilization(uint64_t regst_desc_id)
      : MemoryUtilization(MakeUniqueName(regst_desc_id)) {
    mut_utilization_proto()->mutable_regst_desc_resource()->set_regst_desc_id(
        regst_desc_id);
  }
  ~RegstDescUtilization() = default;

  inline uint64_t regst_desc_id() const {
    return utilization_proto().regst_desc_resource().regst_desc_id();
  }

  void CreateAscendantIfNotFound(UtilizationGraph* graph,
                                 RegstUtilization* leaf) const override;
  static std::string MakeUniqueName(uint64_t regst_desc_id) {
    return std::to_string(regst_desc_id);
  }
};

class RegstUtilization : public MemoryUtilization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstUtilization);
  RegstUtilization(uint64_t regst_desc_id, uint64_t regst_id)
      : MemoryUtilization(MakeUniqueName(regst_desc_id, regst_id)) {
    InitKeyField(regst_desc_id, regst_id);
  }
  explicit RegstUtilization(const UtilizationProto& proto)
      : MemoryUtilization(MakeUniqueName(proto)) {
    InitKeyField(proto.regst_resource().regst_desc_id(),
                 proto.regst_resource().regst_id());
  }
  ~RegstUtilization() = default;

  void InitKeyField(uint64_t regst_desc_id, uint64_t regst_id) {
    auto regst = mut_utilization_proto()->mutable_regst_resource();
    regst->set_regst_desc_id(regst_desc_id);
    regst->set_regst_id(regst_id);
  }

  inline uint64_t regst_desc_id() const {
    return utilization_proto().regst_resource().regst_desc_id();
  }
  inline uint64_t regst_id() const {
    return utilization_proto().regst_resource().regst_id();
  }
  void CreateAscendantIfNotFound(
      UtilizationGraph* graph, RegstUtilization* leaf = nullptr) const override;
  virtual bool IsLeaf() const override { return false; }

  static std::string MakeUniqueName(const UtilizationProto& proto) {
    CHECK(proto.has_regst_resource());
    return MakeUniqueName(proto.regst_resource().regst_desc_id(),
                          proto.regst_resource().regst_id());
  }

  static std::string MakeUniqueName(uint64_t regst_desc_id, uint64_t regst_id) {
    return std::to_string(regst_desc_id) + "," + std::to_string(regst_id);
  }
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_
