#ifndef ONEFLOW_CORE_COMMON_SINGLETON_PTR_H_
#define ONEFLOW_CORE_COMMON_SINGLETON_PTR_H_

namespace oneflow {

namespace private_detail {

template<typename T>
const T* GlobalSingletonPtr() {
  static const T* value = new T();
  return value;
}

}

template<typename T>
const T* SingletonPtr() {
  thread_local const T* value = private_detail::GlobalSingletonPtr<T>();
  return value;
}

}

#endif  // ONEFLOW_CORE_COMMON_SINGLETON_PTR_H_
