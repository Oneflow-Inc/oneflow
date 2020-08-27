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
inline void profile_log(std::ostringstream& ss, const T& str) {
  ss << str;
}

template<typename T, typename... Ts>
inline void profile_log(std::ostringstream& ss, const T& str, const Ts&... rest) {
  ss << str;
  profile_log(ss, rest...);
}

template<typename... Ts>
inline void profile_entry(const Ts&... msgs) {
  std::ostringstream ss;
  profile_log(ss, msgs...);
  LOG(INFO) << "<P>" << ss.str();
}

template<typename... Ts>
inline void profile_end(const Ts&... msgs) {
  std::ostringstream ss;
  profile_log(ss, msgs...);
  LOG(INFO) << "<P>" << ss.str() << "<END>";
}

template<typename... Ts>
inline void profile_entry_if(bool cond, const Ts&... msgs) {
  if (cond) {
    std::ostringstream ss;
    profile_log(ss, msgs...);
    LOG(INFO) << "<P>" << ss.str();
  }
}

template<typename... Ts>
inline void profile_end_if(bool cond, const Ts&... msgs) {
  if (cond) {
    std::ostringstream ss;
    profile_log(ss, msgs...);
    LOG(INFO) << "<P>" << ss.str() << "<END>";
  }
}

}  // namespace

}  // namespace oneflow

#if defined(ENABLE_PROFILE_LOG)

#define PROF(...) profile_entry(__VA_ARGS__)
#define PROFE(...) profile_end(__VA_ARGS__)
#define PROF_IF(cond, ...) profile_entry_if(cond, __VA_ARGS__)
#define PROFE_IF(cond, ...) profile_end_if(cond, __VA_ARGS__)

#else

#define PROF
#define PROFE
#define PROF_IF
#define PROFE_IF

#endif

#endif  // ONEFLOW_CORE_COMMON_PROFILE_H_
