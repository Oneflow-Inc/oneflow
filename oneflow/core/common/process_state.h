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
#ifndef ONEFLOW_CORE_COMMON_PROCESS_STATE_H_
#define ONEFLOW_CORE_COMMON_PROCESS_STATE_H_

#if defined(_MSC_VER)
#include <WinSock2.h>
#include <direct.h>
#include <stdlib.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <unistd.h>
#endif
#include <memory>
#include <string>

namespace oneflow {

inline std::string GetCwd() {
  size_t len = 128;
  std::unique_ptr<char[]> a(new char[len]);
  for (;;) {
    char* p = getcwd(a.get(), len);
    if (p != NULL) {
      return p;
    } else if (errno == ERANGE) {
      len += len;
      a.reset(new char[len]);
    } else {
      return NULL;
    }
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_PROCESS_STATE_H_
