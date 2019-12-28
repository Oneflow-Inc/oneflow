#ifndef ONEFLOW_XRT_TYPES_H_
#define ONEFLOW_XRT_TYPES_H_

#include "oneflow/xrt/types.pb.h"

namespace std {

template<>
struct hash<oneflow::xrt::XrtDevice> {
  size_t operator()(const oneflow::xrt::XrtDevice &device) const {
    return std::hash<int64_t>()(static_cast<int64_t>(device));
  }
};

template<>
struct hash<oneflow::xrt::XrtEngine> {
  size_t operator()(const oneflow::xrt::XrtEngine &engine) const {
    return std::hash<int64_t>()(static_cast<int64_t>(engine));
  }
};

template<>
struct hash<oneflow::xrt::XrtField> {
  size_t operator()(const oneflow::xrt::XrtField &field) const {
    return std::hash<oneflow::xrt::XrtDevice>()(field.device())
           ^ std::hash<oneflow::xrt::XrtEngine>()(field.engine());
  }
};

}  // namespace std

namespace oneflow {
namespace xrt {

constexpr char _XrtLaunchOpType[] = "XrtLaunch";
constexpr char _ArgumentOpType[] = "Argument";
constexpr char _XrtLaunchPrefix[] = "_xrt_launch_";
constexpr char _XrtInArgumentPrefix[] = "_input_argument_";
constexpr char _XrtOutArgumentPrefix[] = "_output_argument_";

constexpr char _ReduceSplitType[] = "ReduceSplit";
constexpr char _ReduceConcatType[] = "ReduceConcat";

constexpr char MutableVariablesAttrName[] = "MutableVariables";
constexpr char IsOptimizerOpAttrName[] = "IsOptimizerOp";
constexpr char TrainPhaseEnabledAttrName[] = "TrainPhaseEnabled";

inline XrtField MakeXrtField(const XrtDevice &device, const XrtEngine &engine) {
  XrtField field;
  field.set_device(device);
  field.set_engine(engine);
  return std::move(field);
}

inline bool operator==(const XrtField &field1, const XrtField &field2) {
  return field1.device() == field2.device() && field1.engine() == field2.engine();
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TYPES_H_
