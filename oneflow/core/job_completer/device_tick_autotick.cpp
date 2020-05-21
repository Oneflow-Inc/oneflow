#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutDeviceTickOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutDeviceTickOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override {
    return op_conf().device_tick_conf().tick_size() > 0;
  }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_device_tick_conf()->add_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kDeviceTickConf, MutDeviceTickOpConTickInputHelper);

}  // namespace oneflow
