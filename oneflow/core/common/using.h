#ifndef ONEFLOW_CORE_COMMON_USING_
#define ONEFLOW_CORE_COMMON_USING_

#include <unordered_set>
#include <unordered_map>
namespace oneflow {

template<typename Key, typename T, typename Hash = std::hash<Key>>
using HashMap = std::unordered_map<Key, T, Hash>;

template<typename Key, typename Hash = std::hash<Key>>
using HashSet = std::unordered_set<Key, Hash>;
#define USING_HASH_CONTAINER

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_USING_
