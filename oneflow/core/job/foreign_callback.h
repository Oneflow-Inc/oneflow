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
#ifndef ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
#define ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_

#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/operator/op_attribute.cfg.h"
#include "oneflow/core/job/job_conf.cfg.h"

namespace oneflow {

class ForeignCallback {
 public:
  ForeignCallback() = default;
  virtual ~ForeignCallback() = default;

  virtual void EagerMirroredCast(const std::shared_ptr<cfg::OpAttribute>& op_attribute,
                                 const std::shared_ptr<cfg::ParallelConf>& parallel_conf) const {
    UNIMPLEMENTED();
  }
  virtual void EagerInterpretCompletedOp(
      const std::shared_ptr<cfg::OpAttribute>& op_attribute,
      const std::shared_ptr<cfg::ParallelConf>& parallel_conf) const {
    UNIMPLEMENTED();
  }

  virtual void OfBlobCall(int64_t unique_id, int64_t ofblob_ptr) const { UNIMPLEMENTED(); }

  virtual void RemoveForeignCallback(int64_t unique_id) const { UNIMPLEMENTED(); }

  // TODO(lixinqi): remove this urgly api after python code migrated into cpp code
  virtual void AddScopeToPyStorage(int64_t scope_symbol_id,
                                   const std::string& scope_proto_str) const {
    UNIMPLEMENTED();
  }

  // return scope_symbol_id
  virtual int64_t MakeScopeSymbol(const std::shared_ptr<cfg::JobConfigProto>& job_conf,
                                  const std::shared_ptr<cfg::ParallelConf>& parallel_conf,
                                  bool is_mirrored) const {
    UNIMPLEMENTED();
    return 0;
  }
  // return parallel_desc_symbol_id
  virtual int64_t MakeParallelDescSymbol(
      const std::shared_ptr<cfg::ParallelConf>& parallel_conf) const {
    UNIMPLEMENTED();
    return 0;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
