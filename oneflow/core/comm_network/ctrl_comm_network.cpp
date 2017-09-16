#include "oneflow/core/comm_network/ctrl_comm_network.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace {

const int32_t max_retry_num = 60;
const int64_t sleep_seconds = 10;

}  // namespace

void CtrlCommNet::Init() {}

}  // namespace oneflow
