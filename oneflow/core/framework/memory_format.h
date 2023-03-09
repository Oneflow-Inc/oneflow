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
#ifndef ONEFLOW_CORE_FRAMEWORK_MEMORY_FORMAT_H_
#define ONEFLOW_CORE_FRAMEWORK_MEMORY_FORMAT_H_
#include <string>
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

enum class MemoryFormatType {
  kContiguous,
  kPreserve,
};

#define MEMORY_FORMAT_SEQ          \
  OF_PP_MAKE_TUPLE_SEQ(Contiguous) \
  OF_PP_MAKE_TUPLE_SEQ(Preserve)

class MemoryFormat final {
 public:
  MemoryFormat(const MemoryFormat&) = default;
  MemoryFormat(MemoryFormat&&) = delete;
  explicit MemoryFormat(MemoryFormatType memory_format_type)
      : memory_format_type_(memory_format_type) {}
  ~MemoryFormat() = default;

  bool operator==(const MemoryFormat& other) const {
    return this->memory_format_type() == other.memory_format_type();
  }

  const std::string& name() const;

  MemoryFormatType memory_format_type() const { return memory_format_type_; }
  static Symbol<MemoryFormat> Get(MemoryFormatType);
#define DECLARE_GET_MEMORY_FORMAT_TYPE_FUNCTION(memory_format_type) \
  static Symbol<MemoryFormat> memory_format_type();
  OF_PP_FOR_EACH_TUPLE(DECLARE_GET_MEMORY_FORMAT_TYPE_FUNCTION, MEMORY_FORMAT_SEQ)
#undef DECLARE_GET_MEMORY_FORMAT_TYPE_FUNCTION

 private:
  MemoryFormatType memory_format_type_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::MemoryFormat> final {
  size_t operator()(const oneflow::MemoryFormat& memory_format) const {
    return static_cast<size_t>(memory_format.memory_format_type());
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_MEMORY_FORMAT_H_
