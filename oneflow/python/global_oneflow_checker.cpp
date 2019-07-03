#include "oneflow/core/job/oneflow.h"

namespace  oneflow {

namespace {

struct GlobalOneflowChecker final {
  GlobalOneflowChecker() = default;
  ~GlobalOneflowChecker() {
    if (Global<Oneflow>::Get() == nullptr) {
      LOG(INFO) << "oneflow exits successfully";
    } else {
      LOG(FATAL) << "Global<Oneflow> is not destroyed yet";
    }
  }
};

GlobalOneflowChecker checker;

}

}
