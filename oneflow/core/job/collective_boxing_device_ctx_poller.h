#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DEVICE_CTX_POLLER_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DEVICE_CTX_POLLER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

namespace boxing {

namespace collective {

class CollectiveBoxingDeviceCtxPoller final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingDeviceCtxPoller);
  ~CollectiveBoxingDeviceCtxPoller();

  void Enqueue(const std::shared_ptr<std::atomic<bool>>& ready_flag, const std::function<void()>&);

 private:
  using EventList = std::list<std::pair<std::shared_ptr<std::atomic<bool>>, std::function<void()>>>;
  friend class Global<CollectiveBoxingDeviceCtxPoller>;
  CollectiveBoxingDeviceCtxPoller();
  std::vector<std::thread> poller_thread_vec_;
  std::vector<std::unique_ptr<std::mutex>> mutex_vec_;
  std::vector<EventList> event_list_vec_;
  std::atomic<bool> shutdown_;
  std::atomic<int64_t> counter_;
  int64_t num_threads_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DEVICE_CTX_POLLER_H_
