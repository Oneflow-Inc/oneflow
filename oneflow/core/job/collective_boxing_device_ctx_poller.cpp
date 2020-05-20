#include "oneflow/core/job/collective_boxing_device_ctx_poller.h"

namespace oneflow {

namespace boxing {

namespace collective {

CollectiveBoxingDeviceCtxPoller::CollectiveBoxingDeviceCtxPoller() : shutdown_(false) {
  poller_thread_ = std::thread([&] {
    while (true) {
      int64_t done_cnt = 0;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_ && event_list_.empty()) { break; }
        for (auto it = event_list_.begin(); it != event_list_.end();) {
          if (*it->first) {
            it->second();
            event_list_.erase(it++);
            done_cnt += 1;
          } else {
            ++it;
          }
        }
      }
      if (done_cnt == 0) { std::this_thread::yield(); }
    }
  });
}

CollectiveBoxingDeviceCtxPoller::~CollectiveBoxingDeviceCtxPoller() {
  shutdown_ = true;
  poller_thread_.join();
}

void CollectiveBoxingDeviceCtxPoller::Enqueue(const std::shared_ptr<std::atomic<bool>>& ready_flag,
                                              const std::function<void()>& callback) {
  CHECK(!shutdown_);
  std::lock_guard<std::mutex> lock(mutex_);
  event_list_.emplace_back(ready_flag, callback);
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
