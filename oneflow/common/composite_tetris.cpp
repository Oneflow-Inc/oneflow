#include <glog/logging.h>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <algorithm>
#include <string>
#include <utility>
#include <memory>
#include <cstdint>
#include "common/composite_tetris.h"

namespace caffe {
// TetrisColumn
template <typename T>
bool TetrisColumn<T>::HasReady() {
  if (ready_ids_.empty()) {
    return false;
  } else {
    return true;
  }
}

template <typename T>
void TetrisColumn<T>::Push(T data, int64_t data_id = -1) {
  data_.push_back(std::make_pair(data_id, data));
  if (!data_map_.count(data_id)) {
    data_map_[data_id] = {};
  }
  auto iter = data_map_.find(data_id);
  iter->second.push_back(std::prev(data_.end()));
  ready_ids_.insert(data_id);
}

template <typename T>
std::map<std::string, std::pair<int64_t, T>> TetrisColumn<T>::Pop() {
  CHECK(!data_.empty());
  std::map<std::string, std::pair<int64_t, T>> item;
  item[name_] = data_.front();
  auto pop_data_id = data_.front().first;
  data_.pop_front();
  ready_ids_.erase(ready_ids_.find(pop_data_id));
  auto iter = data_map_.find(pop_data_id);
  CHECK(iter != data_map_.end());
  CHECK_GT(iter->second.size(), 0);
  iter->second.pop_front();
  if (iter->second.size() == 0) {
    data_map_.erase(iter);
  }

  return item;
}

template <typename T>
std::map<std::string, std::pair<int64_t, T>>
    TetrisColumn<T>::PopDataOfID(int64_t data_id) {
  CHECK(ready_ids_.find(data_id) != ready_ids_.end());
  auto iter = data_map_.find(data_id);
  CHECK(iter != data_map_.end());
  CHECK_GT(iter->second.size(), 0);
  auto pop_data_iter = iter->second.front();
  std::map<std::string, std::pair<int64_t, T>> item;
  item[name_] = *pop_data_iter;
  data_.erase(pop_data_iter);
  iter->second.pop_front();
  if (iter->second.size() == 0) {
    data_map_.erase(iter);
  }
  ready_ids_.erase(ready_ids_.find(item[name_].first));
  return item;
}

template <typename T>
std::multiset<int64_t> TetrisColumn<T>::GetReadyIDs() const {
  return ready_ids_;
}

// FIFOTetris
template <typename T>
bool FIFOTetris<T>::HasReady() {
  if (children_.empty()) {
    return false;
  }
  for (auto kv : children_) {
    if (!(kv.second->HasReady())) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::map<std::string, std::pair<int64_t, T>> FIFOTetris<T>::Pop() {
  std::map<std::string, std::pair<int64_t, T>> items;
  for (auto child_kv : children_) {
    auto child_items = child_kv.second->Pop();
    for (auto item_kv : child_items) {
      items[item_kv.first] = item_kv.second;
    }
  }
  return items;
}

template <typename T>
void FIFOTetris<T>::add(std::shared_ptr<TetrisComponent<T>> child) {
  CHECK(children_.find(child->name()) == children_.end())
    << "child Tetris " << child->name() << " already exist.";
  children_[child->name()] = child;
}

template <typename T>
std::shared_ptr<TetrisComponent<T>>
    FIFOTetris<T>::GetTetrisByName(const std::string& name) const {
  auto it = children_.find(name);
  CHECK(it != children_.end())
    << "Tetris " << name << " is not the child of " << name_ << " .";
  return it->second;
}


// IDTetris
template <typename T>
bool IDTetris<T>::HasReady() {
  UpdateReadyIDs();
  if (ready_ids_.empty()) {
    return false;
  } else {
    return true;
  }
}

template <typename T>
std::map<std::string, std::pair<int64_t, T>> IDTetris<T>::Pop() {
  UpdateReadyIDs();
  CHECK(!ready_ids_.empty());
  return PopDataOfID(*(ready_ids_.begin()));
}

template <typename T>
std::map<std::string, std::pair<int64_t, T>>
    IDTetris<T>::PopDataOfID(int64_t data_id) {
  UpdateReadyIDs();
  CHECK(ready_ids_.find(data_id) != ready_ids_.end());
  std::map<std::string, std::pair<int64_t, T>> items;
  for (auto child_kv : children_) {
    std::map<std::string, std::pair<int64_t, T>> child_items;
    if (child_kv.second->type() == TetrisType::kIDTetris) {
      auto tetris_ptr = dynamic_cast<IDTetris<T>*>(child_kv.second.get());
      child_items = tetris_ptr->PopDataOfID(data_id);
    } else if (child_kv.second->type() == TetrisType::kColumn) {
      auto tetris_ptr = dynamic_cast<TetrisColumn<T>*>(child_kv.second.get());
      child_items = tetris_ptr->PopDataOfID(data_id);
    } else {
      CHECK(false);
    }
    for (auto item_kv : child_items) {
      items[item_kv.first] = item_kv.second;
    }
  }
  ready_ids_.erase(ready_ids_.find(data_id));
  return items;
}

template <typename T>
void IDTetris<T>::add(std::shared_ptr<TetrisComponent<T>> child) {
  bool suitable = child->type() == TetrisType::kIDTetris
    || child->type() == TetrisType::kColumn;
  CHECK(suitable) << "IDTetris can only contain IDTetris and columns.";
  CHECK(children_.find(child->name()) == children_.end())
    << "child Tetris " << child->name() << " already exist.";
  children_[child->name()] = child;
}

template <typename T>
std::shared_ptr<TetrisComponent<T>>
    IDTetris<T>::GetTetrisByName(const std::string& name) const {
  auto it = children_.find(name);
  CHECK(it != children_.end())
    << "Tetris " << name << " is not the child of " << name_ << " .";
  return it->second;
}

template <typename T>
std::multiset<int64_t> IDTetris<T>::GetReadyIDs() {
  UpdateReadyIDs();
  return ready_ids_;
}

template <typename T>
void IDTetris<T>::UpdateReadyIDs() {
  ready_ids_.clear();
  bool first_child = true;
  for (auto child_kv : children_) {
    // get child ready ids
    std::multiset<int64_t> child_ready_ids;
    if (child_kv.second->type() == TetrisType::kIDTetris) {
      auto tetris_ptr = dynamic_cast<IDTetris<T>*>(child_kv.second.get());
      child_ready_ids = tetris_ptr->GetReadyIDs();
    } else if (child_kv.second->type() == TetrisType::kColumn) {
      auto tetris_ptr = dynamic_cast<TetrisColumn<T>*>(child_kv.second.get());
      child_ready_ids = tetris_ptr->GetReadyIDs();
    } else {
      CHECK(false);
    }
    // compute the intersection of ready ids
    if (first_child) {
      ready_ids_ = child_ready_ids;
      first_child = false;
    } else {
      std::multiset<int64_t> tmp_ids;
      std::set_intersection(ready_ids_.begin(), ready_ids_.end(),
        child_ready_ids.begin(), child_ready_ids.end(),
        std::inserter(tmp_ids, tmp_ids.begin()));
      ready_ids_ = tmp_ids;
    }
  }
}

// Instantiate tetris class with int64_t specifications (used for register id).
template class TetrisColumn<int64_t>;
template class FIFOTetris<int64_t>;
template class IDTetris<int64_t>;
}  // namespace caffe
