#ifndef ONEFLOW_UTIL_EVENT_WRITE_HELPER_H_
#define ONEFLOW_UTIL_EVENT_WRITE_HELPER_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

class EventsWriter;

template<DeviceType device_type, typename T>
struct EventWriteHelper {
  static void WritePbToFile(int64_t step, const std::string& value);
  static void WriteScalarToFile(int64_t step, float value, const std::string& tag);
  static void WriteHistogramToFile(int64_t step, const user_op::Tensor& value,
                                   const std::string& tag);
  static void WriteImageToFile(int64_t step, const user_op::Tensor* tensor,
                               const std::string& tag);
};

}  // namespace oneflow

#endif