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
    FOR_RANGE(int32_t, i, 0, parallel_num) {
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      parallel_conf.add_device_name("0:" + std::to_string(i));
      auto unique_op = user_op::UserOpConfWrapperBuilder(op_name + "-unique_" + std::to_string(i))
                           .Op("identity")
                           .Input("in", GenLogicalBlobName(distribute_split_ids_op_conf.name(),
                                                           distribute_split_ids_conf->out(i)))
                           .Output("out")
                           .Build();
      job_builder.AddOps(parallel_conf, {unique_op.op_conf()});

      auto partition_op0 =
          user_op::UserOpConfWrapperBuilder(op_name + "-partition0_" + std::to_string(i))
              .Op("identity")
              .Input("in", unique_op.output("out", 0))
              .Output("out")
              .Build();
      job_builder.AddOps(parallel_conf, {partition_op0.op_conf()});
      sparse_ids.at(0).push_back(partition_op0.output("out", 0));

      auto partition_op1 =
          user_op::UserOpConfWrapperBuilder(op_name + "-partition1_" + std::to_string(i))
              .Op("identity")
              .Input("in", unique_op.output("out", 0))
              .Output("out")
              .Build();
      job_builder.AddOps(parallel_conf, {partition_op1.op_conf()});
      sparse_ids.at(1).push_back(partition_op1.output("out", 0));
    }
    std::vector<std::vector<std::string>> gathered_out(parallel_num);
    FOR_RANGE(int32_t, i, 0, parallel_num) {
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      parallel_conf.add_device_name("0:" + std::to_string(i));
      user_op::UserOpConfWrapperBuilder concat_op_builder(op_name + "-concat_" + std::to_string(i));
      concat_op_builder.Op("concat").Output("out").Attr<int64_t>("axis", 0).Attr<int64_t>(
          "max_dim_size", 10);
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
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      parallel_conf.add_device_name("0:" + std::to_string(i));
      auto concat_gather_op =
          user_op::UserOpConfWrapperBuilder(op_name + "-concat_gather_" + std::to_string(i))
              .Op("concat")
              .Input("in", gathered_out.at(i).at(0))
              .Input("in", gathered_out.at(i).at(1))
              .Output("out")
              .Attr<int64_t>("axis", 0)
              .Attr<int64_t>("max_dim_size", 10)
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
