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
#ifndef ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_
#define ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CopyCommNetOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetOp);
  CopyCommNetOp() = default;
  ~CopyCommNetOp() = default;

  void InitFromOpConf() override;

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_
