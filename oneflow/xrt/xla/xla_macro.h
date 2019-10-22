#ifndef ONEFLOW_XRT_XLA_XLA_MACRO_H_
#define ONEFLOW_XRT_XLA_XLA_MACRO_H_

#define TF_CPP_VLOG_LEVEL_REQUARED(level) \
  "Set env TF_CPP_MIN_VLOG_LEVEL=" #level " to see the details."

#define XRT_STATUS_MACROS_CONCAT_NAME(x, y) \
  XRT_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)
#define XRT_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

#define XRT_CHECK_AND_ASSIGN(lhs, rexpr)                                 \
  XRT_CHECK_AND_ASSIGN_IMPL(                                             \
      XRT_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, \
      rexpr)

#define XRT_CHECK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)                    \
  auto &&statusor = (rexpr);                                               \
  CHECK(statusor.ok()) << xla::WithLogBacktrace(statusor.status()) << ". " \
                       << TF_CPP_VLOG_LEVEL_REQUARED(2);                   \
  lhs = std::move(statusor.ValueOrDie());

#endif  // ONEFLOW_XRT_XLA_XLA_MACRO_H_
