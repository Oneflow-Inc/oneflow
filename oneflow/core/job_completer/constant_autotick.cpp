#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

class MutConstantOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutConstantOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return op_conf().constant_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_constant_conf()->set_tick(lbn);
    return ret;
  }
};

REGISTER_AUTO_TICK(OperatorConf::kConstantConf, MutConstantOpConTickInputHelper);

}  // namespace oneflow
