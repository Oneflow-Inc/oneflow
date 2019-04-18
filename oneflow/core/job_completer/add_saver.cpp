#include "oneflow/core/job_completer/add_saver.h"

namespace oneflow {

void AddSaver(const OpGraph& op_graph, Job* job_conf) {
  JobBuilder builder(job_conf);
  ParallelConf md_save_parallel_conf;
  // only save on master
  md_save_parallel_conf.add_device_name("0:cpu:0");
  md_save_parallel_conf.set_policy(ParallelPolicy::kDataParallel);
  op_graph.ForEachNode([&](const OpNode* node) {
    if (!node->op().op_conf().has_variable_conf()) { return; }
    OperatorConf every_nth_op_conf;
    every_nth_op_conf.set_name("System-Saver-" + node->op().op_name() + "-EveryNth");
    EveryNthOpConf* every_nth_conf = every_nth_op_conf.mutable_every_nth_conf();
    every_nth_conf->set_in(GenLogicalBlobName(node->op().BnInOp2Lbi("out")));
    every_nth_conf->set_out("out");
    const Shape& variable_time_shape = *node->out_blob_time_shape();
    CHECK_GE(variable_time_shape.NumAxes(), 1);
    CHECK_EQ(variable_time_shape.At(0), Global<JobDesc>::Get()->TotalBatchNum());
    every_nth_conf->set_n(
        job_conf->other().predict_conf().tmp_split_fw_bw_train_conf().num_of_batches_in_snapshot()
        * variable_time_shape.Count(1));
    builder.AddOps(node->parallel_desc().parallel_conf(), {every_nth_op_conf});
    OperatorConf model_save_op_conf;
    model_save_op_conf.set_name("System-Saver-" + node->op().op_name() + "-MdSave");
    ModelSaveV2OpConf* model_save_conf = model_save_op_conf.mutable_model_save_v2_conf();
    model_save_conf->set_in(every_nth_op_conf.name() + "/" + every_nth_conf->out());
    model_save_conf->set_lbn(node->op().op_name() + "/"
                             + node->op().op_conf().variable_conf().model_name());
    builder.AddOps(md_save_parallel_conf, {model_save_op_conf});
  });
}

}  // namespace oneflow
