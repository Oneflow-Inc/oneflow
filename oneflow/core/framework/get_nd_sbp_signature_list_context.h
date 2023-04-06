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
#ifndef ONEFLOW_CORE_FRAMEWORK_COMPUTE_GET_ND_SBP_SIGNATURE_LIST_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_COMPUTE_GET_ND_SBP_SIGNATURE_LIST_CONTEXT_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

class Shape;

namespace user_op {

class UserOpDefWrapper;

class GetNdSbpSignatureListContext {
 public:
  virtual ~GetNdSbpSignatureListContext() = default;

  virtual void AddNdSbpSignature(NdSbpSignature&) = 0;
  virtual std::vector<NdSbpSignature>* MutNdSbpSignatureList() = 0;
  virtual const Shape& parallel_hierarchy() = 0;
  virtual const Shape& BlobShape4InputArgNameAndIndex(const std::string& arg_name,
                                                      int32_t index) const = 0;
  template<typename T>
  T Attr(const std::string& attr_name) const {
    return conf_.attr<T>(attr_name);
  }

  const UserOpConfWrapper& user_op_conf() const { return conf_; }

 protected:
  explicit GetNdSbpSignatureListContext(UserOpConfWrapper&& conf) : conf_(std::move(conf)) {}

 private:
  UserOpConfWrapper conf_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_COMPUTE_GET_ND_SBP_SIGNATURE_LIST_CONTEXT_H_
