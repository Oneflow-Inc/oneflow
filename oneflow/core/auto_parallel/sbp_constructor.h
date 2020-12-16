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
#ifndef SBP_CONSTRUCTOR_
#define SBP_CONSTRUCTOR_

#include "sbp_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/operator/normal_model_update_op.h"
#include <fstream>

namespace oneflow {

class SbpConstructor {
 public:
  SbpConstructor() {
    std::ifstream ifs("/home/liyipeng/OneFlow-Benchmark/Classification/cnns/CostRatioFile.txt");
    if (ifs.is_open()) {
      ifs >> CostRatio;
    } else {
      CostRatio = 1e-8;
      std::cout << "CostRatioFile.txt does not exist." << std::endl;
    }
    ifs.close();
    std::cout << "Cost Ratio: " << CostRatio << std::endl;
  };
  ~SbpConstructor() = default;

  void constructSbpGraph(OpGraph& op_graph, Job& job);

  bool OpNodeIsMirrored(OpNode* op_node) const;

  int32_t FindAllMirroredOpNodes(HashMap<std::string, bool>& op_name2is_mirrored,
                                 OpGraph& op_graph);

  void InitializeSbpGraph(OpGraph& op_graph,
                          HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
                          HashMap<std::string, bool>& op_name2is_mirrored,
                          Algorithm::SbpGraph<SbpSignature>& sbp_graph);

  int32_t FindAllFixedOpNodes(HashMap<std::string, bool>& op_name2is_fixed, OpGraph& op_graph);

  Maybe<void> InferLogicalBlobDesc(
      OpGraph& op_graph, const Job& job,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
      HashMap<std::string, bool>& op_name2is_fixed);

  void InferOpNodeSbpSignature(
      OpNode* op_node, const SbpSignature& sbp_sig_conf,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node);

  Maybe<void> InferOpNodeLogicalBlobDesc(OpNode* op_node) const;

  void SplitLogicalInputBlobDesc(OpNode* op_node) const;

  // get Sbp Signature for current op
  // get the way to compute order value in this code
  Maybe<void> InferOpSbpSignature(
      const Operator& op_, const SbpSignature& sbp_sig_conf, const ParallelDesc& parallel_desc,
      const HashMap<std::string, SbpInferHint>& ibn2sbp_infer_hint,
      std::function<Maybe<const OptInt64*>(const std::string&)> BatchAxis4BnInOp,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node);

  Maybe<void> InferSbpSignatureIf(
      const Operator& op_, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node);

  // With sbp signature fixed in upstream, determine a sbp signature for downstream
  Maybe<void> InferSbpSignature(
      const Operator& op_, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node);

  // Compute copy cost.
  void InitializeCopyCost(OpGraph& op_graph,
                          HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
                          HashMap<std::string, bool>& op_name2is_fixed);

  // Compute computation cost for all sbp nodes
  void InitializeComputationCost(
      OpGraph& op_graph, HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
      HashMap<std::string, bool>& op_name2is_fixed);

  // Initialize Cost Model with Sbp from OpGraph
  void StealSbpFromOpGraph(
      OpGraph& op_graph, HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
      HashMap<std::string, bool>& op_name2is_fixed);

  // Update Sbp Signature in each operator
  Maybe<void> UpdateSbpSignature4Op(
      OpGraph& op_graph, Job& job,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
      HashMap<std::string, bool>& op_name2is_fixed);

  // Should customize a function to compute computation cost for each kind of op
  // compute computation cost
  double ComputeComputationCost(const SbpParallel& sbp_parallel_, const BlobDesc& logical_blob_desc,
                                const ParallelDesc& parallel_desc);

  // Algorithm::SbpGraph<SbpSignature> sbp_graph;

  // Time ratio for unit computation cost vs unit copy cost
  double CostRatio;
};

}  // namespace oneflow

#endif  // SBP_CONSTRUCTOR_
