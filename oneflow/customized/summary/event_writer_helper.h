#ifndef ONEFLOW_CUSTOMIZED_SUMMARY_EVENT_WRITER_HELPER_H_
#define ONEFLOW_CUSTOMIZED_SUMMARY_EVENT_WRITER_HELPER_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace summary {

class EventsWriter;

template<DeviceType device_type, typename T>
struct EventWriterHelper {
  static void WritePbToFile(int64_t step, const std::string& value);
  static void WriteScalarToFile(int64_t step, float value, const std::string& tag);
  static void WriteHistogramToFile(int64_t step, const user_op::Tensor& value,
                                   const std::string& tag);
  static void WriteImageToFile(int64_t step, const user_op::Tensor& tensor, const std::string& tag);
};

}  // namespace summary

}  // namespace oneflow

#endif