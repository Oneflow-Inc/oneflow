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
#include <sstream>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job_rewriter/op_graph_pass.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

Maybe<void> DeleteOpAndAncestors(const OpNode* op_node, HashSet<std::string>* to_del_op_names) {
  if (to_del_op_names->find(op_node->op().op_name()) != to_del_op_names->end()) {
    return Maybe<void>::Ok();
  }
  to_del_op_names->insert(op_node->op().op_name());
  for (const auto& bn_in_op : op_node->op().input_bns()) {
    const auto& src_op_node = op_node->SrcNode4Ibn(bn_in_op);
    JUST(DeleteOpAndAncestors(&src_op_node, to_del_op_names));
  }
  return Maybe<void>::Ok();
}

Maybe<std::string> MakeInputOpConf(const std::string& input_op_name,
                                   const InterfaceBlobConf& blob_conf,
                                   OperatorConf* input_op_conf) {
  input_op_conf->set_name(input_op_name);
  auto* input_conf = input_op_conf->mutable_input_conf();
  input_conf->set_out("out");
  input_conf->mutable_blob_conf()->CopyFrom(blob_conf);
  return GenLogicalBlobName(input_op_name, "out");
}

class AddInputOutputOpsPass final : public OpGraphPass {
 public:
  AddInputOutputOpsPass() = default;
  ~AddInputOutputOpsPass() override = default;
  bool IsEnabled() const override { return true; }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

Maybe<void> AddInputOutputOpsPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  // if (!job_builder->job().job_conf().has_signature()) { return Maybe<void>::Ok(); }
  // const auto& job_sig = job_builder->job().job_conf().signature();
  // std::cout << "job signature: " << job_builder->job().job_conf().job_name() << std::endl
  //           << PbMessage2TxtString(job_sig) << std::endl;

  // HashMap<const OpNode*, OperatorConf> replace_op_node2input_op_conf;
  // for (const auto& pair : job_sig.inputs()) {
  //   const auto& input_name = pair.first;
  //   const auto& input_def = pair.second;

  //   CHECK_EQ_OR_RETURN(input_def.consumer_op_names_size(),
  //   input_def.consumer_op_input_bns_size()); FOR_RANGE(int, i, 0,
  //   input_def.consumer_op_names_size()) {
  //     const auto& consumer_op_name = input_def.consumer_op_names(i);
  //     const auto& consumer_ibn = input_def.consumer_op_input_bns(i);
  //     const auto* consumer_op_node = op_graph.OpNode4OpName(consumer_op_name);
  //     std::ostringstream ss;
  //     ss << "consumer_op: " << consumer_op_name << ", input bns: " << std::endl;
  //     for (const auto& ibn : consumer_op_node->op().input_bns()) { ss << "\t" << ibn <<
  //     std::endl; } std::cout << ss.str(); consumer_op_node->op().PrintArgSignature();
  //     consumer_op_node->op().PrintMirroredSignature();
  //     const auto& consumer_lbi = consumer_op_node->op().BnInOp2Lbi(consumer_ibn);
  //     const auto& op_node = consumer_op_node->ProducerOpNode4Lbi(consumer_lbi);
  //     std::shared_ptr<std::string> input_lbn_ptr;
  //     auto iter = replace_op_node2input_op_conf.find(&op_node);
  //     if (iter == replace_op_node2input_op_conf.end()) {
  //       iter = replace_op_node2input_op_conf.emplace(&op_node, OperatorConf()).first;
  //       input_lbn_ptr = JUST(MakeInputOpConf(input_name, input_def.blob_conf(),
  //       &(iter->second)));
  //     }
  //     auto* mut_consumer_op_conf = job_builder->MutableOpConf4OpName(consumer_op_name);
  //     const auto& old_lbn =
  //         ReplaceInputLbnInOpCustomizedConf(mut_consumer_op_conf, consumer_ibn, *input_lbn_ptr);
  //     CHECK_EQ(GenLogicalBlobName(consumer_lbi), old_lbn);
  //   }
  // }

  // std::vector<std::string> to_del_op_names;
  // for (const auto& pair : replace_op_node2input_op_conf) {
  //   job_builder->AddOps(pair.first->parallel_desc().parallel_conf(), {pair.second});
  //   to_del_op_names.push_back(pair.first->op().op_name());
  // }

  // job_builder->DelOps(to_del_op_names);

  ////////////////
  // HashMap<std::string, OperatorConf> op_name2op_conf;
  // HashSet<std::string> ctrl_in_op_names;
  // op_graph.ForEachNode([&](const OpNode* op_node) {
  //   for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
  //     ctrl_in_op_names.insert(ctrl_in_op_name);
  //   }
  // });
  // op_graph.ForEachNode([&](const OpNode* op_node) {
  //   const OperatorConf& op_conf = op_node->op().op_conf();
  //   if (!op_conf.has_cast_to_static_shape_conf()) { return; }
  //   if (!op_conf.ctrl_in_op_name().empty()) { return; }
  //   if (ctrl_in_op_names.find(op_conf.name()) != ctrl_in_op_names.end()) { return; }
  //   if (op_node->in_edges().size() != 1) { return; }
  //   const OpNode* producer = op_node->SoleInEdge()->src_node();
  //   const LogicalBlobId& cast_in_lbi = op_node->op().BnInOp2Lbi("in");
  //   const LogicalBlobId& cast_out_lbi = op_node->op().BnInOp2Lbi("out");
  //   const BlobDesc& cast_in_logical_blob_desc = producer->LogicalBlobDesc4Lbi(cast_in_lbi);
  //   if (cast_in_logical_blob_desc.is_dynamic()) { return; }
  //   for (const OpEdge* out_edge : op_node->out_edges()) {
  //     const OpNode* consumer = out_edge->dst_node();
  //     const std::string& consumer_op_name = consumer->op().op_name();
  //     if (op_name2op_conf.find(consumer_op_name) == op_name2op_conf.end()) {
  //       op_name2op_conf[consumer_op_name] = consumer->op().op_conf();
  //     }
  //     OperatorConf& consumer_op_conf = op_name2op_conf.at(consumer_op_name);
  //     for (const std::string& ibn : consumer->op().input_bns()) {
  //       if (consumer->op().BnInOp2Lbi(ibn) == cast_out_lbi) {
  //         const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn,
  //                                                                 GenLogicalBlobName(cast_in_lbi));
  //         CHECK_EQ(GenLogicalBlobName(cast_out_lbi), old_val);
  //       }
  //     }
  //   }
  //   job_builder->DelOps({op_conf});
  // });
  // for (const auto& pair : op_name2op_conf) { job_builder->MutOpsOnlyOnce({pair.second}); }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_FUNCTION_PASS("AddInputOutputOpsPass", AddInputOutputOpsPass);

}  // namespace oneflow
