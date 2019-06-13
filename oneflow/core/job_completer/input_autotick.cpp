#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutInputOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutInputOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return op_conf().input_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_input_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kInputConf, MutInputOpConTickInputHelper);

}  // namespace oneflow
