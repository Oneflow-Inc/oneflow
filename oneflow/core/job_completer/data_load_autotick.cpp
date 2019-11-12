#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutDataLoadOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutDataLoadOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return op_conf().data_load_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_data_load_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kDataLoadConf, MutDataLoadOpConTickInputHelper);

}  // namespace oneflow
