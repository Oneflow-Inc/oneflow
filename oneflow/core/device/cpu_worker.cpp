#include "oneflow/core/device/cpu_worker.h"

namespace oneflow {

CpuWorker::CpuWorker() {
  thread_ = std::thread([this]() {
    std::function<void()> work;
    while (chan_.Receive(&work) == 0) { work(); }
  });
}

CpuWorker::~CpuWorker() {
  chan_.CloseSendEnd();
  thread_.join();
  chan_.CloseReceiveEnd();
}

}  // namespace oneflow
