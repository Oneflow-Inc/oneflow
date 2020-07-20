#include "oneflow/core/job/collective_boxing_device_ctx_poller.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace boxing {

namespace collective {

CollectiveBoxingDeviceCtxPoller::CollectiveBoxingDeviceCtxPoller() {
  checkpoint2callbacks_.reset(
      new HashMap<CollectiveBoxingDeviceCtxCheckpoint*, std::list<std::function<void()>>>());
  thread_pool_.reset(new ThreadPool(
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf().num_callback_threads()));
  mutex_.reset(new std::mutex());
}

CollectiveBoxingDeviceCtxPoller::~CollectiveBoxingDeviceCtxPoller() {
  checkpoint2callbacks_.reset();
  thread_pool_.reset();
  mutex_.reset();
}

std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint>
CollectiveBoxingDeviceCtxPoller::CreateCheckpoint() {
  auto mutex = mutex_;
  auto checkpoint2callbacks = checkpoint2callbacks_;
  std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> checkpoint(
      new CollectiveBoxingDeviceCtxCheckpoint());
  std::weak_ptr<CollectiveBoxingDeviceCtxCheckpoint> weak_checkpoint(checkpoint);
  auto callback = [mutex, checkpoint2callbacks, weak_checkpoint]() {
    std::list<std::function<void()>> callbacks;
    {
      std::lock_guard<std::mutex> lock(*mutex);
      auto checkpoint_ptr = weak_checkpoint.lock();
      CHECK(checkpoint_ptr);
      auto callbacks_it = checkpoint2callbacks->find(checkpoint_ptr.get());
      CHECK(callbacks_it != checkpoint2callbacks->end());
      callbacks = std::move(callbacks_it->second);
      checkpoint2callbacks->erase(callbacks_it);
    }
    for (const auto& callback : callbacks) { callback(); }
  };
  checkpoint->SetCallback(callback);
  {
    std::lock_guard<std::mutex> lock(*mutex_);
    CHECK(checkpoint2callbacks_->emplace(checkpoint.get(), std::list<std::function<void()>>())
              .second);
  }
  return checkpoint;
}

void CollectiveBoxingDeviceCtxPoller::Enqueue(
    const std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint>& checkpoint,
    const std::function<void()>& callback) {
  {
    std::lock_guard<std::mutex> lock(*mutex_);
    auto it = checkpoint2callbacks_->find(checkpoint.get());
    if (it != checkpoint2callbacks_->end()) {
      it->second.push_back(callback);
      return;
    }
  }
  thread_pool_->AddWork(callback);
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
