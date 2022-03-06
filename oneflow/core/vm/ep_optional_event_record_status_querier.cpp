#include "oneflow/core/vm/ep_device_context.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"

namespace oneflow {

namespace vm {

void EpOptionalEventRecordStatusQuerier::SetLaunched(EpDeviceCtx* device_ctx) {
  CHECK(!launched_);
  if (ep_event_) {
    ep_device_->SetAsActiveDevice();
    device_ctx->stream()->RecordEvent(ep_event_);
  }
  launched_ = true;
}

}

}
