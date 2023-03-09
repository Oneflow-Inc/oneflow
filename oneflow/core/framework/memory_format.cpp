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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/memory_format.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

Symbol<MemoryFormat> MemoryFormat::Get(MemoryFormatType memory_format_type) {
  static const HashMap<MemoryFormatType, Symbol<MemoryFormat>> memory_format_type2memory_format{
#define MAKE_ENTRY(memory_format_type) \
  {OF_PP_CAT(MemoryFormatType::k, memory_format_type), memory_format_type()},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, MEMORY_FORMAT_SEQ)
#undef MAKE_ENTRY
  };
  return memory_format_type2memory_format.at(memory_format_type);
}

const std::string& GetMemoryFormatTypeName(MemoryFormatType memory_format_type) {
  static const HashMap<MemoryFormatType, std::string> memory_format_type2name{
      {MemoryFormatType::kContiguous, "oneflow.contiguous_format"},
      {MemoryFormatType::kPreserve, "oneflow.preserve_format"},
  };
  return memory_format_type2name.at(memory_format_type);
};

const std::string& MemoryFormat::name() const {
  return GetMemoryFormatTypeName(memory_format_type_);
}

#define DEFINE_GET_MEMORY_FORMAT_TYPE_FUNCTION(memory_format_type)                  \
  Symbol<MemoryFormat> MemoryFormat::memory_format_type() {                         \
    static const auto& memory_format =                                              \
        SymbolOf(MemoryFormat(OF_PP_CAT(MemoryFormatType::k, memory_format_type))); \
    return memory_format;                                                           \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_GET_MEMORY_FORMAT_TYPE_FUNCTION, MEMORY_FORMAT_SEQ)
#undef DEFINE_GET_MEMORY_FORMAT_TYPE_FUNCTION

}  // namespace oneflow
