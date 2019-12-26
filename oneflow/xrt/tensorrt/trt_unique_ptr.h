#ifndef ONEFLOW_XRT_TENSORRT_TRT_UNIQUE_PTR_H_
#define ONEFLOW_XRT_TENSORRT_TRT_UNIQUE_PTR_H_

//#include <memory>

namespace oneflow {
namespace xrt {
namespace tensorrt {

namespace nv {

struct PtrDeleter {
  template<typename T>
  inline void operator()(T *obj) {
    if (obj) { obj->destroy(); }
  }
};

template<typename T>
using unique_ptr = std::unique_ptr<T, PtrDeleter>;

}  // namespace nv

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_UNIQUE_PTR_H_
