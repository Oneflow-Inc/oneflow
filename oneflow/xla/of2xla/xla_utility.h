#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_

#include <string>
#include "glog/logging.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"

#include "tensorflow/compiler/xla/util.h"

namespace oneflow {

std::string ExtractOpTypeAsString(const OperatorConf &conf);

std::string BlobName(const LogicalBlobId &lbi);

LogicalBlobId BlobId(const std::string &blob_name);

}  // namespace oneflow

#define ISNULL(x)  nullptr == (x)
#define NOTNULL(x) nullptr != (x)

#define DELETE(x)             \
  do {                        \
    delete x; x = nullptr;    \
  } while (NOTNULL(x))

#define DELETE_V(x)           \
  do {                        \
    delete [] x; x = nullptr; \
  } while (NOTNULL(x))

#define OF_STATUS_MACROS_CONCAT_NAME(x, y) OF_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)
#define OF_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

#define OF_CHECK_AND_ASSIGN(lhs, rexpr) \
  OF_CHECK_AND_ASSIGN_IMPL(             \
      OF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define OF_CHECK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)             \
  auto &&statusor = (rexpr);                                       \
  CHECK(statusor.ok()) << xla::WithLogBacktrace(statusor.status()) \
                       << TF_CPP_VLOG_LEVEL_REQUARED(2);           \
  lhs = std::move(statusor.ValueOrDie());                          \

#define TF_CPP_VLOG_LEVEL_REQUARED(level) \
  "Set env TF_CPP_MIN_VLOG_LEVEL=" #level " to see the details."

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_
