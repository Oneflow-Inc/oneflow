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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_ARG_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_ARG_UTIL_H_

#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/mirrored_parallel.cfg.h"

namespace oneflow {

namespace compatible_py {

class OpArgParallelAttribute {
 public:
  OpArgParallelAttribute(const std::shared_ptr<ParallelDesc>& parallel_desc,
                         const std::shared_ptr<cfg::SbpParallel>& sbp_parallel,
                         const std::shared_ptr<cfg::OptMirroredParallel>& opt_mirrored_parallel)
      : parallel_desc_(parallel_desc),
        sbp_parallel_(sbp_parallel),
        opt_mirrored_parallel_(opt_mirrored_parallel) {
    hash_ = Hash();
  }

  OpArgParallelAttribute(const OpArgParallelAttribute& op_arg_para_attr) = default;
  virtual ~OpArgParallelAttribute() = default;

  std::shared_ptr<ParallelDesc> parallel_desc_symbol() const { return parallel_desc_; }

  std::shared_ptr<cfg::SbpParallel> sbp_parallel() const { return sbp_parallel_; }

  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel() const {
    return opt_mirrored_parallel_;
  }

  bool is_mirrored() const { return opt_mirrored_parallel_->has_mirrored_parallel(); }

  std::size_t _Hash() const { return hash_; }

  bool operator==(const OpArgParallelAttribute& other) const {
    return true && (*parallel_desc_ == *other.parallel_desc_symbol())
           && (*sbp_parallel_ == *other.sbp_parallel())
           && (*opt_mirrored_parallel_ == *other.opt_mirrored_parallel());
  }

  void Assign(const std::shared_ptr<OpArgParallelAttribute>& other) {
    parallel_desc_ = other->parallel_desc_symbol();
    sbp_parallel_ = other->sbp_parallel();
    opt_mirrored_parallel_ = other->opt_mirrored_parallel();
    hash_ = other->_Hash();
  }

  std::string ToString() const {
    return std::string("\nparallel_desc_symbol: ") + parallel_desc_->parallel_conf().DebugString()
           + "\nsbp_parallel: " + sbp_parallel_->DebugString()
           + "\nopt_mirrored_parallel: " + opt_mirrored_parallel_->DebugString() + "\n";
  }

 protected:
  std::size_t Hash() const {
    std::size_t sbp_hash = 0;
    if (!opt_mirrored_parallel_->has_mirrored_parallel()) {
      sbp_hash ^= std::hash<cfg::SbpParallel>()(*sbp_parallel_);
    }
    return sbp_hash ^ (std::hash<ParallelDesc>()(*parallel_desc_))
           ^ (std::hash<cfg::OptMirroredParallel>()(*opt_mirrored_parallel_));
  }

 private:
  std::shared_ptr<ParallelDesc> parallel_desc_;
  std::shared_ptr<cfg::SbpParallel> sbp_parallel_;
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel_;
  std::size_t hash_;
};

}  // namespace compatible_py

}  // namespace oneflow

namespace std {

template<>
struct hash<::oneflow::compatible_py::OpArgParallelAttribute> {
  std::size_t operator()(const ::oneflow::compatible_py::OpArgParallelAttribute& s) const {
    return s._Hash();
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_ARG_UTIL_H_
