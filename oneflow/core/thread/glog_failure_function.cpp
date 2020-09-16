#include "oneflow/core/thread/glog_failure_function.h"

namespace oneflow {

namespace {

void PyFailureFunction() {
  const int ret = Global<GlogFailureFunction>::Get()->RunCallback("panic in C++");
  const bool is_main_thread = Global<GlogFailureFunction>::Get()->IsMainThread();
  if (ret == kMainThreadPanic && is_main_thread) {
    throw MainThreadPanic();
  } else if (ret == kNonMainThreadPanic && is_main_thread == false) {
    throw NonMainThreadPanic();
  } else {
    LOG(ERROR) << "failure callback return code: " << ret << ", is_main_thread: " << is_main_thread
               << ", aborting";
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
