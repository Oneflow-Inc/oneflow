/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <memory>

#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace vm {
class RematableTensorStorage;
}

namespace remat {

class DisjNode {
 public:
  explicit DisjNode(double time) : compute_time_(time), parent_(nullptr), cnt_(1) {}

  bool is_root() { return !bool(parent_); }

  void set_parent(std::shared_ptr<DisjNode>& parent) { parent_ = parent; }
  void set_compute_time(double new_time) { compute_time_ = new_time; }

  void set_cnt(int cnt) { cnt_ = cnt; }
  void add_cnt() { cnt_++; }
  void reduce_cnt() { cnt_--; }

  double compute_time() { return compute_time_; }
  std::shared_ptr<DisjNode> parent() { return parent_; }
  int cnt() { return cnt_; }

  void reset(double t) {
    compute_time_ = t;
    parent_.reset();
  }

 private:
  double compute_time_;
  std::shared_ptr<DisjNode> parent_;
  int cnt_;
};

class DisjointSet {
 public:
  static void merge(std::shared_ptr<DisjNode>& x, std::shared_ptr<DisjNode>& y);
  static std::shared_ptr<DisjNode> find_father(std::shared_ptr<DisjNode>& x);
  static void update_after_compute(vm::RematableTensorStorage* obj);
  static Maybe<void> update_after_release(vm::RematableTensorStorage* obj);
};

}  // namespace remat
}  // namespace oneflow
