#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutModelLoadOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutModelLoadOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return op_conf().model_load_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_model_load_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kModelLoadConf, MutModelLoadOpConTickInputHelper);

}  // namespace oneflow
