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

#include "oneflow/core/common/platform.h"
#ifdef OF_PLATFORM_POSIX
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#endif  // OF_PLATFORM_POSIX

#include "oneflow/core/control/ctrl_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

#ifdef OF_PLATFORM_POSIX

namespace {

sockaddr_in GetSockAddr(const std::string& addr, uint16_t port) {
  sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);
  PCHECK(inet_pton(AF_INET, addr.c_str(), &(sa.sin_addr)) == 1);
  return sa;
}

}  // namespace

int CtrlUtil::FindAvailablePort() const {
  int sock = socket(AF_INET, SOCK_STREAM, 0);

  for (uint16_t port = 10000; port < GetMaxVal<uint16_t>(); ++port) {
    sockaddr_in sa = GetSockAddr("0.0.0.0", port);
    int bind_result = bind(sock, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
    if (bind_result == 0) {
      shutdown(sock, SHUT_RDWR);
      close(sock);
      return port;
    }
  }
  return -1;
}

#else

int CtrlUtil::FindAvailablePort() const { UNIMPLEMENTED(); }

#endif  // OF_PLATFORM_POSIX
}  // namespace oneflow
