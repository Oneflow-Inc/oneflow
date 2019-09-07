#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutDTBOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutDTBOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override {
    return op_conf().define_test_blob_conf().has_tick();
  }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_define_test_blob_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kDefineTestBlobConf, MutDTBOpConTickInputHelper);

}  // namespace oneflow
