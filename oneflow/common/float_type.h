#ifndef ONEFLOW_COMMON_FLOAT_TYPE_H_
#define ONEFLOW_COMMON_FLOAT_TYPE_H_

namespace oneflow {

enum FloatType {
  kFloat,
  kDouble
};

template<FloatType>
size_t GetFloatByteSize();

template<>
inline size_t GetFloatByteSize<kFloat>() {
  return 4;
}

template<>
inline size_t GetFloatByteSize<kDouble>() {
  return 8;
}

} // namespace oneflow

#endif // ONEFLOW_COMMON_FLOAT_TYPE_H_
