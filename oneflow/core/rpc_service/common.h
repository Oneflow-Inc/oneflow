#ifndef ONEFLOW_CORE_RPC_SERVICE_COMMON_H_
#define ONEFLOW_CORE_RPC_SERVICE_COMMON_H_

#include <vector>
#include <string>
#include <cstdint>
#include <msgpack.hpp>

namespace oneflow {
struct PredictParams {
  int64_t tag_id;
  int32_t encode_case;
  int32_t data_type;
  std::string data_id;
  std::string data_name;
  std::vector<std::string> buffers;
  std::string version;
  int32_t max_col;
  std::vector<int64_t> dim_vec;

  MSGPACK_DEFINE(tag_id, encode_case, data_type, data_id, data_name, buffers, version, max_col,
                 dim_vec);
};

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
}  // namespace rpc_service
}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_SERVICE_COMMON_H_