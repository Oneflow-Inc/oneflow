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
#ifndef ONEFLOW_CORE_JOB_PLACEMENT_SCOPE_H_
#define ONEFLOW_CORE_JOB_PLACEMENT_SCOPE_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

class OperatorConf;

class PlacementScope final {
 public:
  PlacementScope(Symbol<ParallelDesc> device_parallel_desc, Symbol<ParallelDesc> host_parallel_desc)
      : device_parallel_desc_(device_parallel_desc), host_parallel_desc_(host_parallel_desc) {}

  size_t hash_value() const { return Hash(device_parallel_desc_, host_parallel_desc_); }

  bool operator==(const PlacementScope& other) const {
    return this->device_parallel_desc_ == other.device_parallel_desc_
           && this->host_parallel_desc_ == other.host_parallel_desc_;
  }

  Symbol<ParallelDesc> device_parallel_desc() const { return device_parallel_desc_; }
  Symbol<ParallelDesc> host_parallel_desc() const { return host_parallel_desc_; }

  Maybe<Symbol<ParallelDesc>> GetParallelDesc(const std::string& device_tag,
                                              const OperatorConf& op_conf) const;

  Maybe<Symbol<ParallelDesc>> GetParallelDesc(const std::string& device_tag,
                                              const std::string& op_type_name) const;

  Maybe<Symbol<ParallelDesc>> GetParallelDesc(const std::string& op_type_name) const;

 private:
  Symbol<ParallelDesc> device_parallel_desc_;
  Symbol<ParallelDesc> host_parallel_desc_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::PlacementScope> final {
  size_t operator()(const oneflow::PlacementScope& val) const { return val.hash_value(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_PLACEMENT_SCOPE_H_
