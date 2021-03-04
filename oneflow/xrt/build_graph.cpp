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
#include "oneflow/xrt/build_graph.h"
#include "oneflow/xrt/api.h"

namespace oneflow {
namespace xrt {

namespace graph_builder {

const Shape& InputTimeShape(const OpNode* op_node) {
  CHECK_NOTNULL(op_node);
  return *(op_node->GetInputBlobFastestTimeShape());
}

const Shape& OutputTimeShape(const OpNode* op_node) {
  CHECK_NOTNULL(op_node);
  return *(op_node->out_blob_time_shape());
}

const SbpParallel& BlobSbpPolicy(const OpNode* op_node, const std::string& name) {
  CHECK_NOTNULL(op_node);
  LogicalBlobId lbi = BlobNameToId(name);
  return op_node->SbpParallel4Lbi(lbi);
}

GraphBuilder::GraphBuilder(const OpGraph* op_graph) : graph_(std::make_shared<XrtGraph>()) {
  op_graph->TopoForEachNode([&](const OpNode* op_node) {
    const Operator* op = &op_node->op();
    XrtNode* node = graph_->AddNode(op->op_conf());
    SetupXrtNode(node, op->op_conf());
    auto& input_output_keys = node_info_[node].input_output_keys;
    for (const std::string& bn : op->output_bns()) {
      std::string output = BlobIdToName(op->BnInOp2Lbi(bn));
      producers_[output] = node;
      input_output_keys[output] = bn;
    }
    for (const std::string& bn : op->input_bns()) {
      std::string input = BlobIdToName(op->BnInOp2Lbi(bn));
      input_output_keys[input] = bn;
      node_info_[node].inputs.insert(input);
    }
    node_info_[node].op_node = op_node;
  });
}

GraphBuilder::GraphBuilder(const XrtLaunchOpConf::Function& function, const DeviceType& device_type,
                           const JobDesc& job_desc)
    : graph_(std::make_shared<XrtGraph>()) {
  for (const auto& arg_conf : function.argument()) {
    XrtNode* node = graph_->AddNode(arg_conf);
    SetupXrtNode(node, arg_conf);
    if (node->IsInArgumentNode()) {
      producers_[arg_conf.value()] = node;
    } else {
      node_info_[node].inputs.insert(arg_conf.value());
    }
    auto& input_output_keys = node_info_[node].input_output_keys;
    input_output_keys = {{arg_conf.value(), "value"}};
  }

  for (const auto& node_conf : function.node()) {
    XrtNode* node = graph_->AddNode(node_conf);
    SetupXrtNode(node, node_conf);
    auto& input_output_keys = node_info_[node].input_output_keys;
    auto op = ConstructOp(node_conf, device_type, &job_desc);
    for (const std::string& bn : op->output_bns()) {
      std::string output = BlobIdToName(op->BnInOp2Lbi(bn));
      producers_[output] = node;
      input_output_keys[output] = bn;
    }
    for (const std::string& bn : op->input_bns()) {
      std::string input = BlobIdToName(op->BnInOp2Lbi(bn));
      input_output_keys[input] = bn;
      node_info_[node].inputs.insert(input);
    }
  }
}

void GraphBuilder::MakeMetaData(const XrtNode* start, const XrtNode* end,
                                const std::string& arg_name, ArgumentMetaData* meta_data) {
  const auto& prod_keys = node_info_.at(start).input_output_keys;
  const auto& cons_keys = node_info_.at(end).input_output_keys;
  meta_data->produce_key = prod_keys.at(arg_name);
  meta_data->consume_key = cons_keys.at(arg_name);
}

void GraphBuilder::BuildGraphEdges() {
  for (const auto& p : node_info_) {
    const XrtNode* node = p.first;
    const util::Set<std::string>& inputs = p.second.inputs;
    for (const std::string& input : inputs) {
      const auto& it = producers_.find(input);
      if (it != producers_.end() && it->second != node) {
        ArgumentMetaData meta;
        MakeMetaData(it->second, node, input, &meta);
        Argument argument(input, meta);
        graph_->Connect(it->second, node, argument);
      }
    }
  }
}

void GraphBuilder::SetupGraphEdges() {
  for (XrtEdge* edge : graph_->Edges()) {
    const OpNode* src = node_info_.at(edge->start()).op_node;
    const OpNode* dst = node_info_.at(edge->end()).op_node;
    const std::string& name = edge->argument().name();

    if (nullptr == src || nullptr == dst) { continue; }
    // Set time shape
    std::vector<Shape> time_shape;
    time_shape.push_back(OutputTimeShape(src));
    time_shape.push_back(InputTimeShape(dst));
    edge->Attr("time_shape", time_shape);
    // Set sbp policy
    std::vector<SbpParallel> sbp_policy;
    sbp_policy.push_back(BlobSbpPolicy(src, name));
    sbp_policy.push_back(BlobSbpPolicy(dst, name));
    edge->Attr("sbp_policy", sbp_policy);
  }
}

std::shared_ptr<XrtGraph> BuildGraph(const XrtLaunchOpConf::Function& function,
                                     const DeviceType& device_type, const JobDesc& job_desc) {
  return GraphBuilder(function, device_type, job_desc).Build();
}

std::shared_ptr<XrtGraph> BuildGraph(const OpGraph* op_graph) {
  return GraphBuilder(op_graph).Build();
}

}  // namespace graph_builder

}  // namespace xrt
}  // namespace oneflow
