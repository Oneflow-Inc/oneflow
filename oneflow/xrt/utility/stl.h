#ifndef ONEFLOW_XRT_UTILITY_STL_H_
#define ONEFLOW_XRT_UTILITY_STL_H_

#include <list>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "google/protobuf/map.h"

namespace oneflow {
namespace xrt {
namespace util {

#define STL_COMP_STRUCT(op, expr)                                          \
  template <typename K>                                                    \
  struct op {                                                              \
    bool operator()(const K &k1, const K &k2) const { return k1 expr k2; } \
  };

STL_COMP_STRUCT(EQ, ==);
STL_COMP_STRUCT(NE, !=);
STL_COMP_STRUCT(GT, >);
STL_COMP_STRUCT(LT, <);
STL_COMP_STRUCT(GE, >=);
STL_COMP_STRUCT(LE, <=);

template <typename T>
using Vector = std::vector<T>;

template <typename T>
using List = std::list<T>;

template <typename K, typename T, typename Hash = std::hash<K>,
          typename Comp = EQ<K>>
using Map = std::unordered_map<K, T, Hash, Comp>;

template <typename K, typename Hash = std::hash<K>, typename Comp = EQ<K>>
using Set = std::unordered_set<K, Hash, Comp>;

template <typename T>
using Stack = std::stack<T>;

template <typename T>
using Queue = std::queue<T>;

template <typename K, typename T>
using PbMap = google::protobuf::Map<K, T>;

template <typename K, typename T>
inline PbMap<K, T> ConvertToPbMap(const Map<K, T> &stdmap) {
  PbMap<K, T> pbmap;
  for (const auto &it : stdmap) {
    pbmap[it.first] = it.second;
  }
  return std::move(pbmap);
}

template <typename T>
using PbVector = google::protobuf::RepeatedPtrField<T>;

}  // namespace util
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_UTILITY_STL_H_
