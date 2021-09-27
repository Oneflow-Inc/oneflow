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

#ifndef ONEFLOW_CORE_MEMORY_MEMORY_CASE_ATTR_UTIL_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_CASE_ATTR_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/attr_value_mem_case.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/infer_output_blob_time_shape_fn_context.h"
#include "oneflow/core/framework/infer_nd_sbp_fn_context.h"

namespace oneflow {

class MemCase {
public:
  MemCase() = default;
  virtual ~MemCase() = default;
  explicit MemCase(const MemoryCase& mem) : mem_case(mem) {}

  MemCase(MemCase const&) = default;
  inline MemCase& operator=(MemCase const& mem){
    if(this == &mem) return *this;
    this->mem_case = mem.mem_case;
    return *this;
  }
  inline MemCase& operator=(MemCase&& mem) noexcept {
    if (this == &mem) {
      return *this;
    }
    this->mem_case = mem.mem_case;
    return *this;
  }
  
  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(mem_case.name_to_attr().at(attr_name));
  }

  template<typename T>
  void SetAttr(const std::string& attr_name, const T value) {
    T& set_val = AttrValueCastNotConst<T>(mem_case.mutable_name_to_attr()->at(attr_name));
    set_val = value;
  }
  
  template<typename T>
  bool HasAttr(const std::string& attr_name) const {
    const auto& iter = mem_case.name_to_attr().find(attr_name);
    if(iter != mem_case.name_to_attr().end()) {return true;}
    return false;
  }

  MemoryCase mem_case;
};


}; // namespace oneflow
#endif  // ONEFLOW_CORE_MEMORY_MEMORY_CASE_ATTR_UTIL_H_
