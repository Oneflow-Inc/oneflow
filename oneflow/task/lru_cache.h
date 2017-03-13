#ifndef _TASK_LRU_CACHE_H_
#define _TASK_LRU_CACHE_H_

#include <stdint.h>
#include <iterator>
#include <list>
#include <unordered_map>
#include <utility>
#include <glog/logging.h>

namespace caffe {
// NOTE(Chonglin): Key type must be unordered_map acceptable type
template <typename K, typename V>
class LRUCache {
 public:
  explicit LRUCache(int32_t capacity) : capacity_(capacity) {}

  bool Get(const K& key, V& value) {
    auto iter = cache_map_.find(key);
    if (iter != cache_map_.end()) {
      cache_list_.splice(cache_list_.begin(), cache_list_, iter->second);
      value = iter->second->second;
      return true;
    }
    return false;
  }

  bool IsFull() const {
    return cache_map_.size() == capacity_;
  }
  void Add(const K& key, const V& value) {
    auto iter = cache_map_.find(key);
    CHECK(iter == cache_map_.end());
    CHECK(!IsFull());
    cache_list_.push_front(CacheNode(key, value));
    cache_map_[key] = cache_list_.begin();
  }
  V RemoveLast() {
    CacheNode last = cache_list_.back();
    cache_map_.erase(last.first);
    cache_list_.pop_back();
    return last.second;
  }

 private:
  using CacheNode = std::pair<K, V>;
  using CacheNodeIter = typename std::list<CacheNode>::iterator;
  std::list<CacheNode> cache_list_;
  std::unordered_map<K, CacheNodeIter> cache_map_;
  int32_t capacity_;
};
}  // namespace caffe
#endif  // _TASK_LRU_CACHE_H_
