#ifndef ONEFLOW_CORE_JOB_MOCK_CALLBACK_NOTIFIER_H_
#define ONEFLOW_CORE_JOB_MOCK_CALLBACK_NOTIFIER_H_

#include "oneflow/core/job/foreign_callback.h"

namespace oneflow {

class MockCallbackNotifier final : public ForeignCallback {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MockCallbackNotifier);
  explicit MockCallbackNotifier(std::function<void()> callback) : callback_(callback) {}

  void Callback() const override { callback_(); }

 private:
  std::function<void()> callback_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MOCK_CALLBACK_NOTIFIER_H_
