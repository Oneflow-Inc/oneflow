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
#include <string>
#include "glog/logging.h"

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/passes/pass.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

namespace {

class ArgMetaDataUpdater {
 public:
  ArgMetaDataUpdater(XrtGraph* graph, const XrtPassOptions& options, const JobDesc* job_desc)
      : graph_(graph), options_(options), job_desc_(job_desc) {}

  void Run() {
    // algorithm::TopologyVisit(*graph, [&](XrtNode *node) {
    for (const XrtNode* node : graph_->Nodes()) {
      util::Map<std::string, std::string> keys;
      if (node->IsArgumentNode()) {
        const auto& conf = *dynamic_cast<const XrtLaunchOpConf::Argument*>(&node->param());
        keys = {{conf.value(), "value"}};
      } else {
        std::shared_ptr<Operator> op = BuildOp(node);
        for (const std::string& bn : op->input_bns()) {
          const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
          keys.emplace(BlobIdToName(lbi), bn);
        }
        for (const std::string& bn : op->output_bns()) {
          const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
          keys.emplace(BlobIdToName(lbi), bn);
        }
      }
      input_output_keys_.emplace(node, std::move(keys));
    }

    for (XrtNode* node : graph_->Nodes()) {
      for (XrtEdge* edge : node->in_edges()) {
        const auto& arg_name = edge->argument().name();
        ArgumentMetaData meta;
        MakeMetaData(edge->start(), node, arg_name, &meta);
        edge->argument().set_meta_data(meta);
      }
      for (XrtEdge* edge : node->out_edges()) {
        const auto& arg_name = edge->argument().name();
        ArgumentMetaData meta;
        MakeMetaData(node, edge->end(), arg_name, &meta);
        edge->argument().set_meta_data(meta);
      }
    }
  }

  void MakeMetaData(const XrtNode* start, const XrtNode* end,
                    const std::string& arg_name,    // NOLINT
                    ArgumentMetaData* meta_data) {  // NOLINT
    const auto& prod_keys = input_output_keys_.at(start);
    const auto& cons_keys = input_output_keys_.at(end);
    meta_data->produce_key = prod_keys.at(arg_name);
    meta_data->consume_key = cons_keys.at(arg_name);
  }

  std::shared_ptr<Operator> BuildOp(const XrtNode* node) {
    DeviceType device_type = XrtDeviceToDeviceType(node->device());
    const auto& conf = *dynamic_cast<const OperatorConf*>(&node->param());
    return ConstructOp(conf, device_type, job_desc_);
  }

  virtual ~ArgMetaDataUpdater() = default;

 private:
  XrtGraph* graph_;
  XrtPassOptions options_;

  const JobDesc* job_desc_;

  util::Map<const XrtNode*, util::Map<std::string, std::string>> input_output_keys_;
};

}  // namespace

class UpdateArgMetaDataPass : public XrtPass {
 public:
  UpdateArgMetaDataPass() = default;

  void Run(XrtGraph* graph, const XrtPassOptions& options,
           const std::vector<Any>& params) override {
    CHECK_GE(params.size(), 1) << "JobDesc is required.";
    const JobDesc* job_desc = any_cast<const JobDesc*>(params[0]);
    ArgMetaDataUpdater(graph, options, job_desc).Run();
  }
};

REGISTER_XRT_PASS(UpdateArgMetaData, UpdateArgMetaDataPass);

}  // namespace xrt
}  // namespace oneflow
