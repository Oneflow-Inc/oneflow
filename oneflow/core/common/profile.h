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
#ifndef ONEFLOW_CORE_COMMON_PROFILE_H_
#define ONEFLOW_CORE_COMMON_PROFILE_H_

namespace oneflow {

namespace {

template<typename T>
inline void trace(std::ostringstream& ss, const T& str) {
  ss << str;
}

template<typename T, typename... Ts>
inline void trace(std::ostringstream& ss, const T& str, const Ts&... rest) {
  ss << str;
  trace(ss, rest...);
}

template<typename... Ts>
inline void trace_entry(const Ts&... msgs) {
  std::ostringstream ss;
  trace(ss, msgs...);
  LOG(INFO) << ss.str();
}

template<typename... Ts>
inline void trace_entry_if(bool cond, const Ts&... msgs) {
  if (cond) {
    std::ostringstream ss;
    trace(ss, msgs...);
    LOG(INFO) << ss.str();
  }
}

}  // namespace

}  // namespace oneflow

#if defined(ENABLE_PROFILE)

#define TRACE(...) trace_entry(__VA_ARGS__)
#define TRACE_IF(cond, ...) trace_entry_if(cond, __VA_ARGS__)
#define PROF(...) TRACE("<P>", __VA_ARGS__)
#define PROFE(...) TRACE("<P>", __VA_ARGS__, "<END>")
#define PROF_IF(cond, ...) TRACE_IF(cond, "<P>", __VA_ARGS__)
#define PROFE_IF(cond, ...) TRACE_IF(cond, "<P>", __VA_ARGS__, "<END")

#else

#define TRACE
#define PROF
#define PROFE
#define PROF_IF
#define PROFE_IF

#endif

#endif  // ONEFLOW_CORE_COMMON_PROFILE_H_
