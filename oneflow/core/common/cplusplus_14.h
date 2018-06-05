#ifndef ONEFLOW_CORE_COMMON_CPLUSPLUS_14_H_
#define ONEFLOW_CORE_COMMON_CPLUSPLUS_14_H_

#if __cplusplus < 201402L

namespace std {

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace std

#endif

#endif  // ONEFLOW_CORE_COMMON_CPLUSPLUS_14_H_
