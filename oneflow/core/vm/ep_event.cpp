#include "oneflow/core/vm/ep_event.h"

namespace oneflow {

EpEvent::EpEvent(ep::Device* device) {
  device_ = device;
  device_->SetAsActiveDevice();
  event_ = device_->CreateEvent();
}

EpEvent::~EpEvent() {
  device_->SetAsActiveDevice();
  device_->DestroyEvent(event_);
}

bool EpEvent::Query() {
  device_->SetAsActiveDevice();
  return CHECK_JUST(event_->QueryDone());
}

}
