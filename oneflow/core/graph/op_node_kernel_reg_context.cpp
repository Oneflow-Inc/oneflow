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
#include "oneflow/core/graph/op_node_kernel_reg_context.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

namespace {

user_op::NaiveTensorDesc GenTensorDescFromBlobDesc(const BlobDesc* blob_desc) {
  user_op::NaiveTensorDesc tensor_desc;
  tensor_desc.set_shape(blob_desc->shape());
  tensor_desc.set_stride(blob_desc->stride());
  tensor_desc.set_data_type(blob_desc->data_type());
  tensor_desc.set_is_dynamic(blob_desc->is_dynamic());
  return tensor_desc;
}

}  // namespace

OpNodeKernelRegContext::OpNodeKernelRegContext(const OpNode* op_node)
    : user_op_conf_(op_node->op().op_conf()) {
  const Operator& op = op_node->op();
  const auto& op_conf = op.op_conf();
  CHECK(op_conf.has_user_conf());

  device_type_ = CHECK_JUST(DeviceType4DeviceTag(op_conf.device_tag()));
  parallel_num_ = op_node->parallel_desc().parallel_num();

  auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                         ArgVec* arg_vec) {
    for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
      for (int32_t i = 0; i < it->second.s_size(); ++i) {
        arg_vec->emplace_back(std::make_pair(it->first, i));
      }
    }
  };
  InitInOrOut(op_conf.user_conf().input(), &inputs_);
  InitInOrOut(op_conf.user_conf().output(), &outputs_);

  {
#define INSERT_TO_ARG2TENSOR_DESC(prefix)                                                \
  for (const auto& bn : op.prefix##_bns()) {                                             \
    const BlobDesc* blob_desc = CHECK_JUST(op.GetLogicalBlobDesc4BnInOp(bn)).get();      \
    if (!blob_desc) { continue; }                                                        \
    arg2tensor_desc_.emplace(GenUnRepeatedBn(bn), GenTensorDescFromBlobDesc(blob_desc)); \
  }

    INSERT_TO_ARG2TENSOR_DESC(input)
    INSERT_TO_ARG2TENSOR_DESC(output)
    // INSERT_TO_ARG2TENSOR_DESC(tmp)

#undef INSERT_TO_ARG2TENSOR_DESC
  }
}

const user_op::TensorDesc* OpNodeKernelRegContext::TensorDesc4ArgNameAndIndex(
    const std::string& arg_name, int32_t index) const {
  auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
  if (it == arg2tensor_desc_.end()) { return nullptr; }
  return &(it->second);
}

const std::shared_ptr<const user_op::AttrVal>& OpNodeKernelRegContext::Attr4Name(
    const std::string& attr_name) const {
  return user_op_conf().Attr4Name(attr_name);
}

}  // namespace oneflow
