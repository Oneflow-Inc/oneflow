#include "oneflow/dl/include/wrapper.h"
#include "oneflow/dl/include/ibv.h"
#include <infiniband/verbs.h>

namespace oneflow {

namespace dl {
DynamicLibrary& getIBVLibrary() {
  static std::string libname = "libibverbs.so";
  static std::string alt_libname = "libibverbs.so.1";
  static DynamicLibrary lib(libname.c_str(), alt_libname.empty() ? nullptr : alt_libname.c_str());
  return lib;
}
}  // namespace dl

namespace ibv {

namespace _stubs {
int ibv_fork_init(void) {
  auto fn = reinterpret_cast<decltype(&ibv_fork_init)>(dl::getIBVLibrary().sym(__func__));
  if (!fn) throw std::runtime_error("Can't get ibv");
  wrapper.ibv_fork_init = fn;
  return fn();
}
}  // namespace _stubs

IBV wrapper = {
#define _REFERENCE_MEMBER(name) _stubs::name,
    IBV_APIS(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
};

}  // namespace ibv
}  // namespace oneflow
