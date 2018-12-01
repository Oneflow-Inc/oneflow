#ifndef ONEFLOW_CORE_RPC_SERVICE_COMMON_H_
#define ONEFLOW_CORE_RPC_SERVICE_COMMON_H_

namespace oneflow {
namespace rpc_service {
enum class result_code : int16_t {
  OK = 0,
  FAIL = 1,
};

enum class error_code {
  OK,
  UNKNOWN,
  FAIL,
  TIMEOUT,
  CANCEL,
  BADCONNECTION,
};

static const size_t MAX_BUF_LEN = 1048576 * 10;
static const size_t HEAD_LEN = 4;
static const size_t PAGE_SIZE = 1024 * 1024;
static const size_t MAX_QUEUE_SIZE = 10240;
}  // namespace rpc_service
}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_SERVICE_COMMON_H_