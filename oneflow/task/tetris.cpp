#include <glog/logging.h>
#include <queue>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <memory>
#include <cstdint>
//#include "context/one.h"
//#include "context/id_map.h"
#include "runtime/comm_bus.h"
//#include "task/node_manager.h"
#include "runtime/event_message.h"
#include "context/id_map.h"
#include "task/tetris.h"

namespace oneflow {
// TetrisFIFOColumn
bool TetrisFIFOColumn::Ready() const {
  return !column_.empty();
}

void TetrisFIFOColumn::Push(int64_t element, int64_t element_id) {
  column_.push(std::make_pair(element, element_id));
}

std::vector<std::pair<int64_t, int64_t>> TetrisFIFOColumn::Pop() {
  CHECK(!column_.empty());
  std::vector<std::pair<int64_t, int64_t>> items;
  auto item = column_.front();
  items.push_back(item);
  column_.pop();
  return items;
}


// TetrisIDColumn
bool TetrisIDColumn::Ready() const {
  return !ready_ids_.empty();
}

void TetrisIDColumn::Push(int64_t element, int64_t element_id) {
  column_[element_id].push_back(element);
  CHECK(column_[element_id].size() <= column_width_)
    << "Tetris simple ID column " << col_id_ << "'s width is limited to "
    << column_width_;
  if (column_[element_id].size() == column_width_) {
    ready_ids_.push(element_id);
  }
}

std::vector<std::pair<int64_t, int64_t>> TetrisIDColumn::Pop() {
  CHECK(!ready_ids_.empty());
  int64_t id = ready_ids_.front();
  ready_ids_.pop();
  auto& elements = column_[id];
  CHECK(elements.size() == column_width_);
  std::vector<std::pair<int64_t, int64_t>> items;
  for (auto element : elements) {
    items.push_back(std::make_pair(element, id));
  }
  column_.erase(id);
  return items;
}

// TetrisSSPColumn
bool TetrisSSPColumn::Ready() const {
  if (data_.empty() || model_.empty()) {
    return false;
  }
  if (staleness_ < 0) {
    return true;
  } else {
    return data_.top().second - newest_ <= staleness_;
  }
}

void TetrisSSPColumn::erase_model(int64_t model) {
  auto it = model_to_version_.find(model);
  CHECK(it != model_to_version_.end());
  auto model_id = it->second;
  model_.erase(it->second);
  model_to_version_.erase(it);

  // auto& id_map = caffe::TheOne<Dtype>::id_map();
  std::shared_ptr<IDMap> id_map(new IDMap());
  auto to_task_id = id_map->task_id_from_register_id(model);

  MsgPtr model_ack_msg(new EventMessage(
    task_id_, to_task_id, model_id, model, MessageType::kConsumed));
  model_ack_msg->set_is_model(true);

  // auto& comm_bus = caffe::TheOne<Dtype>::comm_bus();  // FIXME(jiyuan)
  std::shared_ptr<CommBus> comm_bus(new CommBus(0));
  comm_bus->SendMessage(model_ack_msg);
}

void TetrisSSPColumn::Push(int64_t element, int64_t element_id) {
  // auto& node_manager = caffe::TheOne<Dtype>::node_manager();
  // bool is_model = node_manager->is_in_model_path(element);  // FIXME(jiyuan)
  bool is_model = false;
  if (!is_model) {
    data_.push(std::make_pair(element, element_id));
  } else {
    // If it is the first time got this version(element) of model,
    // then the element is the new model.
    // Otherwise, if this version is already stored,
    // then this push means Model ACK.
    auto it = model_to_version_.find(element);
    if (it == model_to_version_.end()) {
      // New Model
      //if (newest_ < element_id) {
      if (!model_.empty()) {
        auto old = model_.find(newest_);
        CHECK(old != model_.end())
          << "unexpected error in TetrisSSPColumn";
        if (old->second.second == 0) {
          erase_model(old->second.first);
          if (it->second == 0) {
            // can load path be auto-deleted?
            // whether to custom a deleter in load path shared_ptr?
          }
        }
      }
      newest_ = element_id;
      model_[element_id] = std::make_pair(element, 0);
      model_to_version_[element] = element_id;
      // }
    } else {
      // Model ACK
      int64_t version = it->second;
      auto model_it = model_.find(version);
      CHECK(model_it != model_.end());
      CHECK(model_it->second.second > 0)
        << "unexpected error in TetrisSSPColumn";
      model_it->second.second--;
      if (model_it->second.second == 0 && newest_ != version) {
        erase_model(it->first);
      }
    }
  }
}

std::vector<std::pair<int64_t, int64_t>> TetrisSSPColumn::Pop() {
  CHECK(Ready());
  std::vector<std::pair<int64_t, int64_t>> items;
  // Data
  auto data = data_.top();
  data_.pop();

  // Model
  auto newest_model = model_.find(newest_);
  CHECK(newest_model != model_.end())
    << "unexpected error in TetrisSSPColumn";
  std::pair<int64_t, int64_t> model(newest_model->first, newest_);
  newest_model->second.second++;

  items.push_back(data);
  items.push_back(model);
  return items;
}

// TetrisRefCountColumn
bool TetrisRefCountColumn::Ready() const {
  return !avaliable_.empty();
}

void TetrisRefCountColumn::Push(int64_t element, int64_t element_id) {
  ref_count_[element]++;
  CHECK_LE(ref_count_[element], ref_num_);
  if (ref_count_[element] == ref_num_) avaliable_.push(element);
}

std::vector<std::pair<int64_t, int64_t>> TetrisRefCountColumn::Pop() {
  CHECK(Ready());
  std::vector<std::pair<int64_t, int64_t>> items;
  auto item = avaliable_.front();
  items.push_back({ item, -1 });
  ref_count_.erase(item);
  avaliable_.pop();
  return items;
}


// Tetris
bool Tetris::Ready() const {
  for (auto kv : columns_) {
    if (!kv.second->Ready()) {
      return false;
    }
  }
  return true;
}

void Tetris::Push(int32_t col_id, int64_t element, int64_t element_id) {
  auto it = columns_.find(col_id);
  CHECK(it != columns_.end()) << "Column " << col_id << "does not exist.";
  it->second->Push(element, element_id);
}

std::unordered_map<int32_t, std::vector<std::pair<int64_t, int64_t>>>
    Tetris::Pop() {
  std::unordered_map<int32_t, std::vector<std::pair<int64_t, int64_t>>> items;
  for (auto kv : columns_) {
    items[kv.first] = kv.second->Pop();
  }
  return items;
}

void Tetris::Add(std::shared_ptr<TetrisColumn> child) {
  CHECK(columns_.find(child->col_id()) == columns_.end()) << "Column named"
    << child->col_id() << "already exist.";
  columns_[child->col_id()] = child;
}

}  // namespace oneflow
