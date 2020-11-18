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

class GatherDispatchPass final : public JobPass {
 public:
  GatherDispatchPass() = default;
  ~GatherDispatchPass() override = default;
  bool IsEnabled(const JobPassCtx& ctx) const { return true; }
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> GatherDispatchPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name != "gather") { return; }
    user_op::UserOpConfWrapper cur_op(op_conf);
    const std::string& op_name = cur_op.op_name();
    const int64_t parallel_num = op_node->parallel_desc().parallel_num();
    const BlobDesc& in_desc = op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cur_op.input("in", 0)));
    const int64_t num_classes = in_desc.shape().At(0);
    const BlobDesc& indices_desc =
        op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cur_op.input("indices", 0)));
    const int64_t max_dim_size = indices_desc.shape().elem_cnt() * parallel_num;

    OperatorConf distribute_split_ids_op_conf{};
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

    const ParallelDesc parallel_desc(op_node->parallel_desc().parallel_conf());
    FOR_RANGE(int32_t, i, 0, parallel_num) {
      const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(i));
      const int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(i));
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      parallel_conf.add_device_name(std::to_string(machine_id) + ":" + std::to_string(device_id));

      OperatorConf unique_with_counts_op_conf{};
      unique_with_counts_op_conf.set_name(op_name + "-unique_" + std::to_string(i));
      auto* unique_with_counts_conf = unique_with_counts_op_conf.mutable_unique_with_counts_conf();
      unique_with_counts_conf->set_x(GenLogicalBlobName(distribute_split_ids_op_conf.name(),
                                                        distribute_split_ids_conf->out(i)));
      unique_with_counts_conf->set_y("y");
      unique_with_counts_conf->set_idx("idx");
      unique_with_counts_conf->set_count("count");
      unique_with_counts_conf->set_num_unique("num_unique");

      job_builder.AddOps(parallel_conf, {unique_with_counts_op_conf});

      user_op::UserOpConfWrapperBuilder partition_op_builder(op_name + "-partition_"
                                                             + std::to_string(i));
      partition_op_builder.Op("partition")
          .Input("in", GenLogicalBlobName(unique_with_counts_op_conf.name(),
                                          unique_with_counts_conf->y()))
          .Input("in_num_unique", GenLogicalBlobName(unique_with_counts_op_conf.name(),
                                                     unique_with_counts_conf->num_unique()))
          .Attr<int64_t>("parallel_num", parallel_num)
          .Attr<int64_t>("num_classes", num_classes);
      partition_op_builder.Output("out", parallel_num).Output("num_unique", parallel_num);
      auto partition_op = partition_op_builder.Build();
      job_builder.AddOps(parallel_conf, {partition_op.op_conf()});

      FOR_RANGE(int32_t, j, 0, parallel_num) {
        OperatorConf sync_dynamic_resize_op_conf{};
        sync_dynamic_resize_op_conf.set_name(op_name + "-dynamic_resize_" + std::to_string(j)
                                             + std::to_string(i));
        auto* sync_dynamic_resize_conf =
            sync_dynamic_resize_op_conf.mutable_sync_dynamic_resize_conf();
        sync_dynamic_resize_conf->set_in(partition_op.output("out", j));
        sync_dynamic_resize_conf->set_size(partition_op.output("num_unique", j));
        sync_dynamic_resize_conf->set_out("out");
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
      concat_op_builder.Op("concat").Output("out").Attr<int64_t>("axis", 0).Attr<int64_t>(
          "max_dim_size", max_dim_size);
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
                           .Build();
      job_builder.AddOps(parallel_conf, {gather_op.op_conf()});

      // user_op::UserOpConfWrapperBuilder split_like_op_builder(op_name + "-split_like_"
      //                                                        + std::to_string(i));
      // split_like_op_builder.Op("split_like")
      //    .Input("in", gather_op.output("out", 0))
      //    .Attr<int64_t>("axis", 0)
      //    .Output("out", parallel_num);
      //
      // FOR_RANGE(int32_t, j, 0, parallel_num) {
      //  split_like_op_builder.Input("like", sparse_ids.at(i).at(j));
      //}
      // auto split_like_op = split_like_op_builder.Build();
      // job_builder.AddOps(parallel_conf, {split_like_op.op_conf()});
      // gathered_out.at(0).push_back(split_like_op.output("out", 0));
      // gathered_out.at(1).push_back(split_like_op.output("out", 1));
      gathered_out.at(0).push_back(gather_op.output("out", 0));
      gathered_out.at(1).push_back(gather_op.output("out", 0));
    }
    std::vector<std::string> out_list;
    FOR_RANGE(int32_t, i, 0, parallel_num) {
      const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(i));
      const int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(i));
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      parallel_conf.add_device_name(std::to_string(machine_id) + ":" + std::to_string(device_id));
      auto concat_gather_op =
          user_op::UserOpConfWrapperBuilder(op_name + "-concat_gather_" + std::to_string(i))
              .Op("concat")
              .Input("in", gathered_out.at(i).at(0))
              .Input("in", gathered_out.at(i).at(1))
              .Output("out")
              .Attr<int64_t>("axis", 0)
              .Attr<int64_t>("max_dim_size", max_dim_size)
              .Build();
      out_list.push_back(concat_gather_op.output("out", 0));
      job_builder.AddOps(parallel_conf, {concat_gather_op.op_conf()});
    }
    OperatorConf distribute_concat_op_conf{};
    distribute_concat_op_conf.set_name(op_name);
    auto* distribute_concat_conf = distribute_concat_op_conf.mutable_distribute_concat_conf();
    FOR_RANGE(int32_t, i, 0, parallel_num) { distribute_concat_conf->add_in(out_list.at(i)); }
    distribute_concat_conf->set_axis(0);
    distribute_concat_conf->set_out("out_0");
    job_builder.MutOpsOnlyOnce({distribute_concat_op_conf});
    // job_builder.AddOps(op_node->parallel_desc().parallel_conf(), {distribute_concat_op_conf});
    // job_builder->DelOps({cur_op.op_conf()});
    // auto end_op = user_op::UserOpConfWrapperBuilder(op_name)
    //                  .Op("identity")
    //                  .Input("in", GenLogicalBlobName(distribute_concat_op_conf.name(),
    //                                                  distribute_concat_conf->out()))
    //                  .Output("out")
    //                  .Build();
    // job_builder.MutOpsOnlyOnce({end_op.op_conf()});
    LOG(INFO) << "debugstr" << job_builder.job().DebugString();
  });
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("GatherDispatchPass", GatherDispatchPass);

}  // namespace oneflow
