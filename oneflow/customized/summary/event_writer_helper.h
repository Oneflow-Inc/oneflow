/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
