#include <memory>

#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace vm {
class TensorStorage;
}

namespace dtr {

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
  static void update_after_compute(vm::TensorStorage* obj);
  static Maybe<void> update_after_evict(vm::TensorStorage* obj);
};

}  // namespace dtr
}  // namespace oneflow
