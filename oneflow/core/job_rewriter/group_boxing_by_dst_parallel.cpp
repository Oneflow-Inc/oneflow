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
#include "oneflow/core/job_rewriter/group_boxing_by_dst_parallel.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"
#define COVER_BY_B_

namespace oneflow {

void GroupBoxingByDstParallel(const OpGraph& op_graph, JobBuilder* job_builder) {
  // mapping from a logical blob id to a hashmap.
  // The mapped hashmap maps from different placement of memory blocks to a vector of downstream
  // operators Collect relationship between logical blob id and SBP Parallel at different downstream
  // nodes
  HashMap<LogicalBlobId, HashMap<std::pair<ParallelDesc, SbpParallel>,
                                 std::vector<std::pair<const OpNode*, std::string>>>>
      lbi2consumer_grouped_by_parallel_sbp;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  op_graph.ForEachNode([&](const OpNode* node) {
    OperatorConf::OpTypeCase op_type_case = node->op().op_conf().op_type_case();
    if (IsClassRegistered<int32_t, DisableInputBoxingGroup>(op_type_case)) { return; }
    for (const std::string& ibn : node->op().input_bns()) {
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      const OpNode& producer = node->ProducerOpNode4Lbi(lbi);
      const SbpParallel& producer_sbp = producer.SbpParallel4Lbi(lbi);
      const SbpParallel& consumer_sbp = node->SbpParallel4BnInOp(ibn);
#ifdef COVER_BY_B_
      // We actually don't have any copy cost for an upstream opeartor with Broadcast SBPParallel.
      if (producer_sbp.has_broadcast_parallel()) continue;
#endif  // COVER_BY_B_
      // if we have downstream placement different from upstream placement
      if (producer.parallel_desc() != node->parallel_desc() || producer_sbp != consumer_sbp) {
        // put the pair of node and input blob name into a grouped vector
        lbi2consumer_grouped_by_parallel_sbp[lbi][{node->parallel_desc(), consumer_sbp}].push_back(
            {node, ibn});
        // maps from operator node to operator configuration
        if (op_node2op_conf.find(node) == op_node2op_conf.end()) {
          op_node2op_conf[node] = node->op().op_conf();
        }
      }
    }
  });
#ifdef COVER_BY_B_
  // Use broadcast in the only proxy under some condition.
  for (auto& lbi7groups : lbi2consumer_grouped_by_parallel_sbp) {
    const LogicalBlobId& lbi = lbi7groups.first;
    HashMap<std::pair<ParallelDesc, SbpParallel>,
            std::vector<std::pair<const OpNode*, std::string>>>& ParallelPairs2NodeBlobs =
        lbi7groups.second;
    if (ParallelPairs2NodeBlobs.size() < 2) { continue; }
    // Suppose all the parallel num is the same for different groups
    const ParallelDesc& dst_parallel_desc = ParallelPairs2NodeBlobs.begin()->first.first;
    // Create a new broadcast sbp parallel
    SbpParallel dst_sbp_parallel;
    dst_sbp_parallel.mutable_broadcast_parallel();
    // make them pairs
    std::pair<ParallelDesc, SbpParallel> ParallelPairs(dst_parallel_desc, dst_sbp_parallel);
    // to decide if we transfer to a proxy with SBP Paralell Broadcast
    bool transfer2B;
    auto search_B_parallel = ParallelPairs2NodeBlobs.find(ParallelPairs);
    if (search_B_parallel != ParallelPairs2NodeBlobs.end()) {
      // If we find any B as SBP Parallel in downstream, we should directly use B as SBP of a proxy.
      transfer2B = true;
    } else {
      // test debug
      // Check if hashmap works
      for (auto& parallel7group : ParallelPairs2NodeBlobs) {
        const SbpParallel& dst_sbp_parallel = parallel7group.first.second;
        if (dst_sbp_parallel.has_broadcast_parallel()) {
          // should replace print out message with assert.
          std::cout << "Different broadcast type" << std::endl;
        }
      }
      // If we have a lot of sbp patterns in downstream, suppose we have n SBP patterns and N
      // devices. Then the copy cost from upstream under any SBP Parallel to multiple
      // downstreams upstream -> multiple downstream >= n * min(copy cost) >=
      // n * CopyCost(S(i)->S(j)) >= N * CopyCost(S(i)->S(j)) = CopyCost(S(i)->B)
      // = upstream -> proxy(B) -> multiple downstream
      transfer2B = ParallelPairs2NodeBlobs.size() >= dst_parallel_desc.parallel_num();
    }

    if (transfer2B) {
      std::vector<std::pair<const OpNode*, std::string>>& dst_node_blobs =
          ParallelPairs2NodeBlobs[ParallelPairs];
      for (auto it = ParallelPairs2NodeBlobs.begin(); it != ParallelPairs2NodeBlobs.end();) {
        if (it->first == ParallelPairs) {
          it++;
        } else {
          // erase the other SBP patterns
          dst_node_blobs.insert(dst_node_blobs.end(), it->second.begin(), it->second.end());
          it = ParallelPairs2NodeBlobs.erase(it);
        }
      }
    }
  }
#endif  // COVER_BY_B_
  // Why 7???
  for (const auto& lbi7groups : lbi2consumer_grouped_by_parallel_sbp) {
    const LogicalBlobId& lbi = lbi7groups.first;
    for (const auto& parallel7group : lbi7groups.second) {
      // just move on if we don't have two same placement pattern
      if (parallel7group.second.size() < 2) { continue; }
      const ParallelDesc& dst_parallel_desc = parallel7group.first.first;
      const SbpParallel& dst_sbp_parallel = parallel7group.first.second;
      // Insert an operator proxy to store a blob under a specific SBP.
      OperatorConf identity_op_conf{};
      identity_op_conf.set_name("System-Boxing-Identity-" + NewUniqueId());
      IdentityOpConf* identity_conf = identity_op_conf.mutable_identity_conf();
      identity_conf->set_in(GenLogicalBlobName(lbi));
      identity_conf->set_out("out");
      job_builder->AddOps(dst_parallel_desc.parallel_conf(), {identity_op_conf});
      // Assign the specific SBP for the identity op
      SbpSignature identity_sbp_signature;
      (*identity_sbp_signature.mutable_bn_in_op2sbp_parallel())["in"] = dst_sbp_parallel;
      (*identity_sbp_signature.mutable_bn_in_op2sbp_parallel())["out"] = dst_sbp_parallel;
      (*job_builder->mutable_job_parallel_view_conf()
            ->mutable_op_name2sbp_signature_conf())[identity_op_conf.name()] =
          identity_sbp_signature;
      LogicalBlobId grouped_lbi;
      grouped_lbi.set_op_name(identity_op_conf.name());
      grouped_lbi.set_blob_name(identity_conf->out());
      // Replace the old input blob with a new one for the downstream operators
      for (const auto& consumer7ibn : parallel7group.second) {
        const OpNode* consumer = consumer7ibn.first;
        const std::string& ibn = consumer7ibn.second;
        OperatorConf& consumer_op_conf = op_node2op_conf[consumer];
        const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn,
                                                                GenLogicalBlobName(grouped_lbi));
        CHECK_EQ(GenLogicalBlobName(lbi), old_val);
      }
    }
  }
  for (const auto& op_node7op_conf : op_node2op_conf) {
    job_builder->MutOpsOnlyOnce({op_node7op_conf.second});
  }
}

}  // namespace oneflow
