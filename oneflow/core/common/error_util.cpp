#include "oneflow/core/common/error_util.h"

namespace oneflow {

Error ErrorUtil::Ok() { return Error(); }

Error ErrorUtil::ProtoParseFailedError(const std::string& msg) {
  Error error;
  error.set_msg(msg);
  error.mutable_proto_parse_failed_error();
  return error;
}

}  // namespace oneflow
