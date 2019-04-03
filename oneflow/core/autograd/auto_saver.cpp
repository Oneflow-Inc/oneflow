#include "oneflow/core/autograd/auto_saver.h"

namespace oneflow {

void AutoSaver(const OpGraph& op_graph, JobConf1* job_conf) {
  JobConfBuilder builder(job_conf);
  ParallelConf md_save_parallel_conf;
  // only save on master
  md_save_parallel_conf.add_device_name("0:CPU:0");
  op_graph.ForEachNode([&](const OpNode* node) {
    if (!node->op().op_conf().has_variable_conf()) { return; }
    OperatorConf every_nth_op_conf;
    every_nth_op_conf.set_name("System-Saver-" + node->op().op_name() + "-EveryNth");
    EveryNthOpConf* every_nth_conf = every_nth_op_conf.mutable_every_nth_conf();
    every_nth_conf->set_in(GenLogicalBlobName(node->op().BnInOp2Lbi("out")));
    every_nth_conf->set_out("out");
    every_nth_conf->set_n(
        job_conf->other().predict_conf().tmp_split_fw_bw_train_conf().num_of_batches_in_snapshot());
    builder.AddOps(node->parallel_desc().parallel_conf(), {every_nth_op_conf});
    OperatorConf model_save_op_conf;
    model_save_op_conf.set_name("System-Saver-" + node->op().op_name() + "-MdSave");
    ModelSaveV2OpConf* model_save_conf = model_save_op_conf.mutable_model_save_v2_conf();
    model_save_conf->set_in(every_nth_op_conf.name() + "/" + every_nth_conf->out());
    model_save_conf->set_lbn(GenLogicalBlobName(node->op().BnInOp2Lbi("out")));
    builder.AddOps(md_save_parallel_conf, {model_save_op_conf});
  });
}

}  // namespace oneflow
