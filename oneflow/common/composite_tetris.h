#ifndef _COMMON_COMPOSITE_TETRIS_H_
#define _COMMON_COMPOSITE_TETRIS_H_
#include <glog/logging.h>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <memory>
#include <cstdint>
#include <unordered_map>

namespace caffe {
/*
 * Here we implement the Tetris Model by using Composite Pattern (Component,
 * Composite, Leaf).
 *
 * Tetris Model has 1 kind column as Leaf, which stores data, data id, etc.
 * (you can set data id = -1 or remain default if you don't need it)
 *
 * And Tetris Model has 2 kinds of Composite, whose duty is to organize columns
 * in different ways:
 * 1.the Composite synchronise data columns in it with same data id, which
 *   means it will consider the id when call hasReady() or Pop();
 * 2.the Composite don't synchronise columns, just use FIFO to Pop().
 *   
 * (1) IDTetris can only contain IDTetris and columns. Not FIFOTetris.
 * (2) FIFOTetris can contain FIFOTetris, IDTetris and columns.
 * (3) All data are stored in columns
 */

enum class TetrisType {
  kUnknown = 0,
  kColumn,
  kFIFOTetris,
  kIDTetris
};

// Component
template <typename T>
class TetrisComponent {
 public:
  explicit TetrisComponent(TetrisType type) : type_(type) {}
  TetrisComponent(const std::string& name, TetrisType type)
    : name_(name), type_(type) {}
  virtual ~TetrisComponent() {}

  virtual bool HasReady() = 0;
  virtual std::map<std::string, std::pair<int64_t, T>> Pop() = 0;

  void set_name(const std::string& name) { name_ = name; }
  std::string name() const { return name_; }
  TetrisType type() const { return type_; }
 protected:
  std::string name_;
  TetrisType type_{ TetrisType::kUnknown };
};

// Leaf
template <typename T>
class TetrisColumn : public TetrisComponent<T> {
 public:
  TetrisColumn() : TetrisComponent(TetrisType::kColumn) {}
  explicit TetrisColumn(const std::string& name)
    : TetrisComponent(name, TetrisType::kColumn) {}
  virtual ~TetrisColumn() {}
  bool HasReady() override;
  void Push(T data, int64_t data_id = -1);
  std::map<std::string, std::pair<int64_t, T>> Pop() override;
  std::map<std::string, std::pair<int64_t, T>> PopDataOfID(int64_t data_id);
  std::multiset<int64_t> GetReadyIDs() const;
 private:
  using DataListType = std::list<std::pair<int64_t, T>>;
  using DataListTypeIter = typename DataListType::iterator;
  DataListType data_;
  // NOTE(Chonglin): maintain the iterator of each data_id, so that we don't
  // need to iterate over whole list to delete the record with some data_id
  std::unordered_map<int64_t, std::list<DataListTypeIter>> data_map_;
  std::multiset<int64_t> ready_ids_;
  // std::map<int64_t, T> data_;
};

// FIFO Composite
template <typename T>
class FIFOTetris : public TetrisComponent<T> {
 public:
  FIFOTetris() : TetrisComponent(TetrisType::kFIFOTetris) {}
  explicit FIFOTetris(const std::string& name)
    : TetrisComponent(name, TetrisType::kFIFOTetris) {}
  virtual ~FIFOTetris() {}
  bool HasReady() override;
  std::map<std::string, std::pair<int64_t, T>> Pop() override;
  void add(std::shared_ptr<TetrisComponent<T>> child);
  std::shared_ptr<TetrisComponent<T>> GetTetrisByName(const std::string& name)
    const;
 private:
  std::map<std::string, std::shared_ptr<TetrisComponent<T>>> children_;
};

// ID Composite
template <typename T>
class IDTetris : public TetrisComponent<T> {
 public:
  IDTetris() : TetrisComponent(TetrisType::kIDTetris) {}
  explicit IDTetris(const std::string& name)
    : TetrisComponent(name, TetrisType::kIDTetris) {}
  virtual ~IDTetris() {}
  bool HasReady() override;
  // IDTetris'Pop() will be ordered by data_id
  std::map<std::string, std::pair<int64_t, T>> Pop() override;
  std::map<std::string, std::pair<int64_t, T>> PopDataOfID(int64_t data_id);
  void add(std::shared_ptr<TetrisComponent<T>> child);
  std::shared_ptr<TetrisComponent<T>> GetTetrisByName(const std::string& name)
    const;
  std::multiset<int64_t> GetReadyIDs();
 private:
  void UpdateReadyIDs();
  std::map<std::string, std::shared_ptr<TetrisComponent<T>>> children_;
  std::multiset<int64_t> ready_ids_;
};

}  // namespace caffe
#endif  // _COMMON_COMPOSITE_TETRIS_H_
