#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutLrShedulerOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutLrShedulerOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool IsTickInputBound() const override { return op_conf().lr_sheduler_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_lr_sheduler_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kLrShedulerConf, MutLrShedulerOpConTickInputHelper);

}  // namespace oneflow
