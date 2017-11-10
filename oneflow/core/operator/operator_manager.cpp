#include "oneflow/core/operator/operator_manager.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

HashMap<int, std::function<Operator*()>>& OpTypeCase2Creator() {
  static HashMap<int, std::function<Operator*()>> obj;
  return obj;
}

}  // namespace

std::shared_ptr<Operator> OpMgr::AddOp(const OperatorConf& op_conf) {
  std::shared_ptr<Operator> ret = ConstructOp(op_conf);
  op_list_.emplace_back(ret);
  return ret;
}

void OpMgr::AllOpToProto(PbRpf<OperatorProto>* ret) {
  ret->Clear();
  for (auto it = op_list_.begin(); it != op_list_.end();) {
    if (std::shared_ptr<const Operator> op = it->lock()) {
      op->ToProto(ret->Add());
      ++it;
    } else {
      op_list_.erase(it++);
    }
  }
}

std::shared_ptr<Operator> OpMgr::ModelUpdateOp() {
  if (!model_update_op_) {
    OperatorConf mdupdt_conf;
    mdupdt_conf.set_name("model_update");
    if (JobDesc::Singleton()->is_train()) {
      const TrainConf& train_conf =
          JobDesc::Singleton()->job_conf().train_conf();
      if (train_conf.has_normal_mdupdt_conf()) {
        *(mdupdt_conf.mutable_normal_mdupdt_conf()) =
            train_conf.normal_mdupdt_conf();
      } else if (train_conf.has_momentum_mdupdt_conf()) {
        *(mdupdt_conf.mutable_momentum_mdupdt_conf()) =
            train_conf.momentum_mdupdt_conf();
      } else if (train_conf.has_rmsprop_mdupdt_conf()) {
        *(mdupdt_conf.mutable_rmsprop_mdupdt_conf()) =
            train_conf.rmsprop_mdupdt_conf();
      } else if (train_conf.has_lars_mdupdt_conf()) {
        *(mdupdt_conf.mutable_lars_mdupdt_conf()) =
            train_conf.lars_mdupdt_conf();
      } else {
        UNEXPECTED_RUN();
      }
    } else if (JobDesc::Singleton()->is_predict()) {
      mdupdt_conf.mutable_normal_mdupdt_conf();
    } else {
      UNEXPECTED_RUN();
    }
    model_update_op_ = AddOp(mdupdt_conf);
  }
  return model_update_op_;
}

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*()> creator) {
  CHECK(OpTypeCase2Creator().emplace(op_type_case, creator).second);
}

Operator* CreateOp(OperatorConf::OpTypeCase op_type_case) {
  return OpTypeCase2Creator().at(op_type_case)();
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf) {
  std::shared_ptr<Operator> ret(CreateOp(op_conf.op_type_case()));
  ret->InitFromOpConf(op_conf);
  return ret;
}

}  // namespace oneflow
