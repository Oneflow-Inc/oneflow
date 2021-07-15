/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_XRT_XLA_XLA_MACRO_H_
#define ONEFLOW_XRT_XLA_XLA_MACRO_H_

#define TF_CPP_VLOG_LEVEL_REQUARED(level) \
  "Set env TF_CPP_MIN_VLOG_LEVEL=" #level " to see the details."

#define MOLA_STATUS_MACROS_CONCAT_NAME(x, y) MOLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)
#define MOLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

#define MOLA_CHECK_AND_ASSIGN(lhs, rexpr)                                                        \
  MOLA_CHECK_AND_ASSIGN_IMPL(MOLA_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, \
                             rexpr)

#define MOLA_CHECK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)                   \
  auto&& statusor = (rexpr);                                               \
  CHECK(statusor.ok()) << xla::WithLogBacktrace(statusor.status()) << ". " \
                       << TF_CPP_VLOG_LEVEL_REQUARED(2);                   \
  lhs = std::move(statusor.ValueOrDie());

#endif  // ONEFLOW_XRT_XLA_XLA_MACRO_H_
