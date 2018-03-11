#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
namespace oneflow {

static std::shared_ptr<const Operator> NormalMdUpdtCompTaskNode::ConstructModelUpdateOp(int32_t in_num) {
  OperatorConf op_conf;
  op_conf.set_name("normal_md_update_" + NewUniqueId());
  NormalModelUpdateOpConf* mdupdt_conf = op_conf.mutable_normal_mdupdt_conf();
  const JobDesc* job_desc = JobDesc::Singleton();
  if (job_desc->IsTrain()) {
    *(mdupdt_conf->mutable_user_conf()) =
        job_desc->job_conf().train_conf().normal_model_update_conf();
  }
  mdupdt_conf->set_in_num(in_num);
  return ConstructOp(op_conf);
}

void NormalMdUpdtCompTaskNode::BindInRegst() {
  size_t ibn_idx = 0;
  ExecNode* node = mut_exec_gph().SoleNode();
  for (const auto& pair : consumed_regsts()) {
    node->BindBnInOpAndRegst(node->op()->input_bns().at(ibn_idx++),
                             pair.second.lock());
  }
}




}  // namespace oneflow
