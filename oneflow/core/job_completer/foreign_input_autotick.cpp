#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutForeignInputOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutForeignInputOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override {
    return op_conf().foreign_input_conf().has_tick();
  }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_foreign_input_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kForeignInputConf, MutForeignInputOpConTickInputHelper);

}  // namespace oneflow
