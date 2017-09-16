#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/snode.h"
#include "oneflow/core/schedule/utilization.pb.h"
#include "oneflow/core/schedule/utilization_util.h"

#define COMPUTATION_UTILIZATION_TYPE_SEQ                                \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kComputation,               \
                       ComputationUtilization)                          \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kDevComputation,            \
                       DeviceComputationUtilization)                    \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kStream, StreamUtilization) \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kTask, TaskUtilization)     \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kTaskStream, TaskStreamUtilization)

#define MEMORY_UTILIZATION_TYPE_SEQ                                           \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kMemory, MemoryUtilization)       \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kDevMemory,                       \
                       DeviceMemoryUtilization)                               \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kRegstDesc, RegstDescUtilization) \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationResource::kRegst, RegstUtilization)

#define UTILIZATION_TYPE_SEQ \
  COMPUTATION_UTILIZATION_TYPE_SEQ MEMORY_UTILIZATION_TYPE_SEQ

#define UTILIZATION_EVENT_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(TaskStreamUtilization) \
  OF_PP_MAKE_TUPLE_SEQ(RegstUtilization)

namespace oneflow {
namespace schedule {

class UtilizationGraph;

class Utilization : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Utilization);
  virtual ~Utilization() = default;

  virtual uint32_t ParallelNum(const UtilizationGraph&) const = 0;
  void Reduce(const UtilizationGraph&);
  void CreateAscendantIfNotFound(UtilizationGraph* ugraph) const;

  float GetTimePerBatch(const UtilizationGraph& ugraph) const;

  UtilizationResource::ResourceTypeCase GetResourceTypeCase() {
    return utilization_proto().resource().resource_type_case();
  }

  std::string VisualStr() const;
  //	getter
  inline const UtilizationProto& utilization_proto() const {
    return utilization_proto_;
  }
  inline const std::list<const UtilizationProto*>& raw_protos() const {
    return raw_protos_;
  }

  virtual std::string type() const = 0;

  //	setter
  inline std::list<const UtilizationProto*>* mut_raw_protos() {
    return &raw_protos_;
  }

 protected:
  explicit Utilization(const UtilizationResource& resource)
      : SNode(UtilizationUtil::GetUniqueName(resource)) {
    *utilization_proto_.mutable_resource() = resource;
  }
  inline UtilizationProto* mut_utilization_proto() {
    return &utilization_proto_;
  }

 private:
  UtilizationProto utilization_proto_;
  std::list<const UtilizationProto*> raw_protos_;
};

#define DECLARE_UTILIZATION(type_case, class_name) class class_name;
OF_PP_FOR_EACH_TUPLE(DECLARE_UTILIZATION, UTILIZATION_TYPE_SEQ)

template<typename U>
struct GetUtilizationResourceTypeCase;

#define SPECIALIZE_UTILIZATION_RESOURCE_TYPE(type_case, class_name)         \
  template<>                                                                \
  struct GetUtilizationResourceTypeCase<class_name> {                       \
    static const UtilizationResource::ResourceTypeCase resource_type_case = \
        type_case;                                                          \
  };

OF_PP_FOR_EACH_TUPLE(SPECIALIZE_UTILIZATION_RESOURCE_TYPE, UTILIZATION_TYPE_SEQ)

#define UTILIZATION_BOILERPLATE(class_name, base_class)                     \
  static const UtilizationResource::ResourceTypeCase resource_type_case =   \
      GetUtilizationResourceTypeCase<class_name>::resource_type_case;       \
  OF_DISALLOW_COPY_AND_MOVE(class_name);                                    \
  class_name(const UtilizationResource& resource) : base_class(resource) {} \
  ~class_name() = default;                                                  \
  std::string type() const override { return __CLASS_NAME__; }

class ComputationUtilization : public Utilization {
 public:
  UTILIZATION_BOILERPLATE(ComputationUtilization, Utilization);
  uint32_t ParallelNum(const UtilizationGraph& ugraph) const override;
};

class DeviceComputationUtilization : public ComputationUtilization {
 public:
  UTILIZATION_BOILERPLATE(DeviceComputationUtilization, ComputationUtilization);

  uint32_t ParallelNum(const UtilizationGraph& ugraph) const override;
};

class StreamUtilization : public ComputationUtilization {
 public:
  UTILIZATION_BOILERPLATE(StreamUtilization, ComputationUtilization);
  uint32_t ParallelNum(const UtilizationGraph&) const override { return 1u; }
  float GetInitiationInterval(const UtilizationGraph& ugraph) const {
    return GetTimePerBatch(ugraph);
  }
};

class TaskUtilization : public ComputationUtilization {
 public:
  UTILIZATION_BOILERPLATE(TaskUtilization, ComputationUtilization);
  inline uint64_t task_id() const {
    return utilization_proto().resource().task().task_id();
  }
  uint32_t ParallelNum(const UtilizationGraph& ugraph) const override;
  float GetDuration(const UtilizationGraph& ugraph) const {
    return GetTimePerBatch(ugraph);
  }
};

class TaskStreamUtilization : public ComputationUtilization {
 public:
  UTILIZATION_BOILERPLATE(TaskStreamUtilization, ComputationUtilization);
  uint32_t ParallelNum(const UtilizationGraph&) const override { return 1u; }
};

class MemoryUtilization : public Utilization {
 public:
  UTILIZATION_BOILERPLATE(MemoryUtilization, Utilization);

  uint32_t ParallelNum(const UtilizationGraph& ugraph) const override;
};

class DeviceMemoryUtilization : public MemoryUtilization {
 public:
  UTILIZATION_BOILERPLATE(DeviceMemoryUtilization, MemoryUtilization);
};

class RegstDescUtilization : public MemoryUtilization {
 public:
  UTILIZATION_BOILERPLATE(RegstDescUtilization, MemoryUtilization);
};

class RegstUtilization : public MemoryUtilization {
 public:
  UTILIZATION_BOILERPLATE(RegstUtilization, MemoryUtilization);
  uint32_t ParallelNum(const UtilizationGraph&) const override { return 1u; }
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_H_
