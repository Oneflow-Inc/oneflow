#ifndef ONEFLOW_CORE_COMMON_SYM_H_
#define ONEFLOW_CORE_COMMON_SYM_H_

namespace oneflow {

template<typename T>
class Sym final {
 public:

  T* operator->();
  T operator*();
  operator T();
  operator uint64_t();
  
};

}

#endif  // ONEFLOW_CORE_COMMON_SYM_H_
