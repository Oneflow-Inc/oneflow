#include <unique_ptr>

namespace oneflow {
namespace xrt {
namespace tensorrt {

namespace nv {

struct PtrDeleter {
  template <typename T>
  void operator()(T *obj) {
    if (obj) {
      obj->destory();
    }
  }
};

template <typename T>
using unique_ptr = std::unique_ptr<T, PtrDeleter>;

}  // namespace nv

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
