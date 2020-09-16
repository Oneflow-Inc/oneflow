#include "oneflow/core/thread/glog_failure_function.h"

namespace oneflow {

namespace {

void PyFailureFunction() {
  int ret = Global<GlogFailureFunction>::Get()->RunCallback("panic in C++");
  if (ret == kMainThreadPanic) {
    throw MainThreadPanic();
  } else if (ret == kNonMainThreadPanic) {
    throw NonMainThreadPanic();
  } else {
    abort();
  }
}

}  // namespace

void GlogFailureFunction::SetCallback(const py_failure_callback& f) {
  failure_function_ = f;
  is_function_set_ = true;
  UpdateThreadLocal();
}

void GlogFailureFunction::UpdateThreadLocal() {
  if (is_function_set_) { google::InstallFailureFunction(&PyFailureFunction); }
}

}  // namespace oneflow
