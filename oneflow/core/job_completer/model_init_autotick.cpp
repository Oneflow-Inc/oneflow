#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutModelInitOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutModelInitOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return op_conf().model_init_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_model_init_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kModelInitConf, MutModelInitOpConTickInputHelper);

}  // namespace oneflow
