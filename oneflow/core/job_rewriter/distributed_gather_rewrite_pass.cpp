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
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

class DistributedGatherRewritePass final : public JobPass {
 public:
  DistributedGatherRewritePass() = default;
  ~DistributedGatherRewritePass() override = default;
  bool IsEnabled(const JobPassCtx& ctx) const { return true; }
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> DistributedGatherRewritePass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const int64_t parallel_num = op_node->parallel_desc().parallel_num();
    if (parallel_num == 1) { return; }
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name != "distributed_gather") { return; }
    user_op::UserOpConfWrapper cur_op(op_conf);
    const SbpParallel& indices_sbp =
        op_node->SbpParallel4Lbi(GenLogicalBlobId(cur_op.input("indices", 0)));
    const SbpParallel& in_sbp = op_node->SbpParallel4Lbi(GenLogicalBlobId(cur_op.input("in", 0)));
    if (!indices_sbp.has_split_parallel() || !in_sbp.has_split_parallel()) { return; }
    if (indices_sbp.split_parallel().axis() != 0 || in_sbp.split_parallel().axis() != 0) { return; }
    const std::string& op_name = cur_op.op_name();
    const BlobDesc& in_desc = op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cur_op.input("in", 0)));
    const int64_t num_classes = in_desc.shape().At(0);
    const BlobDesc& indices_desc =
        op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cur_op.input("indices", 0)));
    const int64_t max_dim_size = indices_desc.shape().elem_cnt();
    OperatorConf distribute_split_ids_op_conf{};
    distribute_split_ids_op_conf.set_scope_symbol_id(op_conf.scope_symbol_id());

    distribute_split_ids_op_conf.set_name(op_name + "-distribute_split_ids");
    auto* distribute_split_ids_conf = distribute_split_ids_op_conf.mutable_distribute_split_conf();
    distribute_split_ids_conf->set_in(cur_op.input("indices", 0));
    distribute_split_ids_conf->set_axis(0);

    FOR_RANGE(int32_t, i, 0, parallel_num) {
      const std::string& blob_name = "out_" + std::to_string(i);
      distribute_split_ids_conf->add_out(blob_name);
    }
    job_builder.AddOps(op_node->parallel_desc().parallel_conf(), {distribute_split_ids_op_conf});

    OperatorConf distribute_split_data_op_conf{};
    distribute_split_data_op_conf.set_scope_symbol_id(op_conf.scope_symbol_id());

    distribute_split_data_op_conf.set_name(op_name + "-distribute_split_data");
    auto* distribute_split_data_conf =
        distribute_split_data_op_conf.mutable_distribute_split_conf();
    distribute_split_data_conf->set_in(cur_op.input("in", 0));
    distribute_split_data_conf->set_axis(0);
    FOR_RANGE(int32_t, i, 0, parallel_num) {
      const std::string& blob_name = "out_" + std::to_string(i);
      distribute_split_data_conf->add_out(blob_name);
    }
    job_builder.AddOps(op_node->parallel_desc().parallel_conf(), {distribute_split_data_op_conf});

    std::vector<std::vector<std::string>> sparse_ids(parallel_num);
    std::vector<std::string> map_ids;

    const ParallelDesc parallel_desc(op_node->parallel_desc().parallel_conf());
    FOR_RANGE(int32_t, i, 0, parallel_num) {
      const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(i));
      const int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(i));
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      parallel_conf.add_device_name(std::to_string(machine_id) + ":" + std::to_string(device_id));

      user_op::UserOpConfWrapperBuilder gather_dispatch_op_builder(op_name + "-gather_dispatch_"
                                                                   + std::to_string(i));
      gather_dispatch_op_builder.Op("gather_dispatch")
          .Input("indices", GenLogicalBlobName(distribute_split_ids_op_conf.name(),
                                               distribute_split_ids_conf->out(i)))
          .Output("idx")
          .Attr<int64_t>("parallel_num", parallel_num)
          .Attr<int64_t>("num_classes", num_classes)
          .ScopeSymbolId(op_conf.scope_symbol_id());
      gather_dispatch_op_builder.Output("out", parallel_num).Output("count", parallel_num);
      auto gather_dispatch_op = gather_dispatch_op_builder.Build();
      job_builder.AddOps(parallel_conf, {gather_dispatch_op.op_conf()});
      map_ids.push_back(gather_dispatch_op.output("idx", 0));

      FOR_RANGE(int32_t, j, 0, parallel_num) {
        OperatorConf sync_dynamic_resize_op_conf{};
        sync_dynamic_resize_op_conf.set_scope_symbol_id(op_conf.scope_symbol_id());

        sync_dynamic_resize_op_conf.set_name(op_name + "-dynamic_resize_" + std::to_string(j)
                                             + std::to_string(i));
        auto* sync_dynamic_resize_conf =
            sync_dynamic_resize_op_conf.mutable_sync_dynamic_resize_conf();
        sync_dynamic_resize_conf->set_in(gather_dispatch_op.output("out", j));
        sync_dynamic_resize_conf->set_size(gather_dispatch_op.output("count", j));
        sync_dynamic_resize_conf->set_out("out");
        sync_dynamic_resize_conf->set_axis(0);
        job_builder.AddOps(parallel_conf, {sync_dynamic_resize_op_conf});

        sparse_ids.at(j).push_back(GenLogicalBlobName(sync_dynamic_resize_op_conf.name(),
                                                      sync_dynamic_resize_conf->out()));
      }
    }
    std::vector<std::vector<std::string>> gathered_out(parallel_num);
    FOR_RANGE(int32_t, i, 0, parallel_num) {
      const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(i));
      const int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(i));
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      parallel_conf.add_device_name(std::to_string(machine_id) + ":" + std::to_string(device_id));
      user_op::UserOpConfWrapperBuilder concat_op_builder(op_name + "-concat_" + std::to_string(i));
      concat_op_builder.Op("concat")
          .Output("out")
          .Attr<int64_t>("axis", 0)
          .Attr<int64_t>("max_dim_size", max_dim_size)
          .ScopeSymbolId(op_conf.scope_symbol_id());
      FOR_RANGE(int32_t, j, 0, parallel_num) {
        concat_op_builder.Input("in", sparse_ids.at(i).at(j));
      }
      auto concat_op = concat_op_builder.Build();
      job_builder.AddOps(parallel_conf, {concat_op.op_conf()});

      auto gather_op = user_op::UserOpConfWrapperBuilder(op_name + "-gather_" + std::to_string(i))
                           .Op("gather")
                           .Input("in", GenLogicalBlobName(distribute_split_data_op_conf.name(),
                                                           distribute_split_data_conf->out(i)))
                           .Input("indices", concat_op.output("out", 0))
                           .Attr<int64_t>("axis", 0)
                           .Output("out")
                           .ScopeSymbolId(op_conf.scope_symbol_id())
                           .Build();
      job_builder.AddOps(parallel_conf, {gather_op.op_conf()});

      user_op::UserOpConfWrapperBuilder split_like_op_builder(op_name + "-split_like_"
                                                              + std::to_string(i));
      split_like_op_builder.Op("split_like")
          .Input("in", gather_op.output("out", 0))
          .Attr<int64_t>("axis", 0)
          .Output("out", parallel_num)
          .ScopeSymbolId(op_conf.scope_symbol_id());

      FOR_RANGE(int32_t, j, 0, parallel_num) {
        split_like_op_builder.Input("like", sparse_ids.at(i).at(j));
      }
      auto split_like_op = split_like_op_builder.Build();
      job_builder.AddOps(parallel_conf, {split_like_op.op_conf()});
      FOR_RANGE(int32_t, j, 0, parallel_num) {
        gathered_out.at(j).push_back(split_like_op.output("out", j));
      }
    }
    std::vector<std::string> out_list;
    FOR_RANGE(int32_t, i, 0, parallel_num) {
      const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(i));
      const int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(i));
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      parallel_conf.add_device_name(std::to_string(machine_id) + ":" + std::to_string(device_id));
      auto concat_gather_op_builder =
          user_op::UserOpConfWrapperBuilder(op_name + "-concat_gather_" + std::to_string(i));
      concat_gather_op_builder.Op("concat")
          .Output("out")
          .Attr<int64_t>("axis", 0)
          .Attr<int64_t>("max_dim_size", max_dim_size)
          .ScopeSymbolId(op_conf.scope_symbol_id());
      FOR_RANGE(int32_t, j, 0, parallel_num) {
        concat_gather_op_builder.Input("in", gathered_out.at(i).at(j));
      }
      auto concat_gather_op = concat_gather_op_builder.Build();
      job_builder.AddOps(parallel_conf, {concat_gather_op.op_conf()});

      auto gather_map_op =
          user_op::UserOpConfWrapperBuilder(op_name + "-gather_map_" + std::to_string(i))
              .Op("gather")
              .Input("in", concat_gather_op.output("out", 0))
              .Input("indices", map_ids.at(i))
              .Attr<int64_t>("axis", 0)
              .Output("out")
              .ScopeSymbolId(op_conf.scope_symbol_id())
              .Build();
      job_builder.AddOps(parallel_conf, {gather_map_op.op_conf()});
      out_list.push_back(gather_map_op.output("out", 0));
    }
    OperatorConf distribute_concat_op_conf{};
    distribute_concat_op_conf.set_scope_symbol_id(op_conf.scope_symbol_id());
    distribute_concat_op_conf.set_name(op_name);
    auto* distribute_concat_conf = distribute_concat_op_conf.mutable_distribute_concat_conf();
    FOR_RANGE(int32_t, i, 0, parallel_num) { distribute_concat_conf->add_in(out_list.at(i)); }
    distribute_concat_conf->set_axis(0);
    distribute_concat_conf->set_out("out_0");
    job_builder.MutOpsOnlyOnce({distribute_concat_op_conf});
  });
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("DistributedGatherRewritePass", DistributedGatherRewritePass);

}  // namespace oneflow
