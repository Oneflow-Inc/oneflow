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
#ifndef ONEFLOW_CORE_OPERATOR_OP_CONF_SYMBOL_H_
#define ONEFLOW_CORE_OPERATOR_OP_CONF_SYMBOL_H_

#include <string>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/op_conf.cfg.h"

namespace oneflow {

class OperatorConfSymbol final {
 public:
  OperatorConfSymbol(const OperatorConfSymbol&) = delete;
  OperatorConfSymbol(OperatorConfSymbol&&) = delete;
  OperatorConfSymbol(int64_t symbol_id, const OperatorConf& op_conf);

  ~OperatorConfSymbol() = default;

  const OperatorConf& op_conf() const { return op_conf_; }
  const Maybe<int64_t>& symbol_id() const { return symbol_id_; }
  std::shared_ptr<cfg::OperatorConf> data() const { return data_; }

 private:
  Maybe<int64_t> symbol_id_;
  OperatorConf op_conf_;
  std::shared_ptr<cfg::OperatorConf> data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OP_CONF_SYMBOL_H_
