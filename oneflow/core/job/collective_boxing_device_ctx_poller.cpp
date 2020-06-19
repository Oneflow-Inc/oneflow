#include "oneflow/core/job/collective_boxing_device_ctx_poller.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace boxing {

namespace collective {

CollectiveBoxingDeviceCtxPoller::CollectiveBoxingDeviceCtxPoller() : shutdown_(false), counter_(0) {
  num_threads_ =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf().num_callback_threads();
  mutex_vec_.reserve(num_threads_);
  for (int64_t tid = 0; tid < num_threads_; ++tid) { mutex_vec_.emplace_back(new std::mutex()); }
  poller_thread_vec_.resize(num_threads_);
  event_list_vec_.resize(num_threads_);
  for (int64_t tid = 0; tid < num_threads_; ++tid) {
    poller_thread_vec_.at(tid) = std::thread([tid, this] {
      std::mutex& mutex = *mutex_vec_.at(tid);
      EventList& event_list = event_list_vec_.at(tid);
      EventList local_event_list;
      while (true) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          local_event_list.splice(local_event_list.end(), event_list);
          if (local_event_list.empty() && shutdown_) { break; }
        }
        int64_t done_cnt = 0;
        for (auto it = local_event_list.begin(); it != local_event_list.end();) {
          if (*it->first) {
            it->second();
            local_event_list.erase(it++);
            done_cnt += 1;
          } else {
            ++it;
          }
        }
        if (done_cnt == 0) { std::this_thread::yield(); }
      }
    });
  }
}

CollectiveBoxingDeviceCtxPoller::~CollectiveBoxingDeviceCtxPoller() {
  shutdown_ = true;
  for (int64_t tid = 0; tid < num_threads_; ++tid) { poller_thread_vec_.at(tid).join(); }
}

void CollectiveBoxingDeviceCtxPoller::Enqueue(const std::shared_ptr<std::atomic<bool>>& ready_flag,
                                              const std::function<void()>& callback) {
  CHECK(!shutdown_);
  int64_t tid = counter_.fetch_add(1) % num_threads_;
  std::lock_guard<std::mutex> lock(*mutex_vec_.at(tid));
  event_list_vec_.at(tid).emplace_back(ready_flag, callback);
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
