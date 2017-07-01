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

namespace oneflow {

std::string GetCwd() {
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
