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
#ifndef ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {

namespace user_op {

class TensorDesc;

class UserOpSbpSignatureBuilder final {
 public:
  UserOpSbpSignatureBuilder(SbpSignatureList* sbp_sig_list) : sbp_sig_list_(sbp_sig_list) {}

  UserOpSbpSignatureBuilder& Split(const OpArg& op_arg, int64_t axis);
  UserOpSbpSignatureBuilder& Split(const std::vector<OpArg>& op_args, int64_t axis);
  UserOpSbpSignatureBuilder& Split(const std::vector<std::pair<std::string, int32_t>>& op_args,
                                   int64_t axis);

  UserOpSbpSignatureBuilder& Broadcast(const OpArg& op_arg);
  UserOpSbpSignatureBuilder& Broadcast(const std::vector<OpArg>& op_args);
  UserOpSbpSignatureBuilder& Broadcast(const std::vector<std::pair<std::string, int32_t>>& op_args);

  UserOpSbpSignatureBuilder& PartialSum(const OpArg& op_arg);
  UserOpSbpSignatureBuilder& PartialSum(const std::vector<OpArg>& op_args);
  UserOpSbpSignatureBuilder& PartialSum(
      const std::vector<std::pair<std::string, int32_t>>& op_args);

  void Build() { *(sbp_sig_list_->mutable_sbp_signature()->Add()) = sbp_sig_tmp_; }

 private:
  SbpSignatureList* sbp_sig_list_;
  SbpSignature sbp_sig_tmp_;
};

class SbpContextBase {
 public:
  SbpContextBase() = default;
  virtual ~SbpContextBase() = default;

  virtual const TensorDesc& LogicalTensorDesc4InputArgNameAndIndex(
      const std::string& input_arg_name, int32_t index) const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  virtual DeviceType device_type() const = 0;
  virtual int64_t parallel_num() const = 0;

  template<typename T>
  T Attr(const std::string& attr_name) const {
    return user_op_conf().attr<T>(attr_name);
  }
  virtual const UserOpConfWrapper& user_op_conf() const = 0;
};

class SbpContext : public SbpContextBase {
 public:
  SbpContext() = default;
  ~SbpContext() override = default;

  virtual UserOpSbpSignatureBuilder NewBuilder() = 0;
};

class InferSbpSignatureFnContext : public SbpContextBase {
 public:
  InferSbpSignatureFnContext() = default;
  ~InferSbpSignatureFnContext() override = default;

  virtual SbpSignature* mutable_sbp_signature() = 0;
  virtual const SbpSignature& sbp_signature_conf() const = 0;
  virtual const SbpParallel& SbpParallelHint4InputArgNameAndIndex(const std::string& input_arg_name,
                                                                  int32_t index) const = 0;
};

struct GetSbpFnUtil {
  static Maybe<void> DefaultBroadcastToBroadcast(SbpContext*);
  static Maybe<void> SplitForEachAxis(SbpContext*);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
