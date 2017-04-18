#ifndef ONEFLOW_TASK_FSM_TETRIS_H_
#define ONEFLOW_TASK_FSM_TETRIS_H_
#include <glog/logging.h>
#include <queue>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <memory>
#include <cstdint>

namespace oneflow {
/*
 * We implement a simple Tetris model without recursive composition which is
 * more suitable for thor. So that we can achieve a more simple and efficient
 * solution.
 * Tetris contains SimpleIDColumn, SimpleFIFOColumn, SimpleSSPColumn, etc.
 */
template <typename T>
class TetrisColumn {
 public:
  explicit TetrisColumn(int32_t col_id) : col_id_(col_id) {}
  virtual ~TetrisColumn() {}

  virtual bool Ready() const = 0;
  virtual void Push(T element, int64_t element_id) = 0;
  virtual std::vector<std::pair<T, int64_t>> Pop() = 0;
  int32_t col_id() const { return col_id_; }
 protected:
  int32_t col_id_;
};

template <typename T>
class TetrisFIFOColumn :public TetrisColumn<T> {
 public:
  explicit TetrisFIFOColumn(int32_t col_id) :
     TetrisColumn(col_id) {}
  virtual ~TetrisFIFOColumn() {}

  bool Ready() const override;
  void Push(T element, int64_t element_id = -1) override;
  std::vector<std::pair<T, int64_t>> Pop() override;
 private:
  std::queue<std::pair<T, int64_t>> column_;
};

template <typename T>
class TetrisIDColumn :public TetrisColumn<T> {
 public:
  explicit TetrisIDColumn(int32_t col_id) :
    TetrisColumn(col_id) {}
  TetrisIDColumn(int32_t col_id, uint32_t width) :
    TetrisColumn(col_id), column_width_(width) {}
  virtual ~TetrisIDColumn() {}

  bool Ready() const override;
  // |element_id| must be set for the IDColumn
  void Push(T element, int64_t element_id) override;
  std::vector<std::pair<T, int64_t>> Pop() override;

  void set_column_width(uint32_t width) { column_width_ = width; }
  uint32_t column_width() const { return column_width_; }
 private:
  uint32_t column_width_{ 0 };
  std::unordered_map<int64_t, std::vector<T>> column_;
  std::queue<int64_t> ready_ids_;
};

template <typename T>
class TetrisSSPColumn :public TetrisColumn<T> {
 public:
  TetrisSSPColumn(int32_t col_id, int32_t staleness, int64_t task_id) :
  TetrisColumn(col_id), staleness_(staleness), newest_(LLONG_MIN),
  task_id_(task_id) {}
  virtual ~TetrisSSPColumn() {}

  bool Ready() const override;
  void Push(T element, int64_t element_id) override;
  std::vector<std::pair<T, int64_t>> Pop() override;

 private:
  // Data
  using DataPair = std::pair<T, int64_t>;
  struct CmpGreater {
    bool operator ()(const DataPair & a, const DataPair & b) {
      return a.second > b.second;
    }
  };

  std::priority_queue<DataPair, std::vector<DataPair>, CmpGreater> data_;
  // Model
  int32_t staleness_;  // -1 means unlimited staleness
  using ModelCount = std::pair<T, int32_t>;
  std::unordered_map<int64_t, ModelCount> model_;
  std::unordered_map<T, int64_t> model_to_version_;
  int64_t newest_;

  int64_t task_id_;
  void erase_model(T model);
};

template <typename T>
class TetrisRefCountColumn :public TetrisColumn<T> {
 public:
  explicit TetrisRefCountColumn(int32_t col_id, uint32_t ref_num) :
    TetrisColumn(col_id), ref_num_(ref_num) {
    CHECK_GT(ref_num, 0);
  }
  virtual ~TetrisRefCountColumn() {}
  bool Ready() const override;
  // |element_id| is useless for RefCountColumn, always -1
  void Push(T element, int64_t element_id = -1) override;
  std::vector<std::pair<T, int64_t>> Pop() override;
 private:
  uint32_t ref_num_;
  std::unordered_map<T, uint32_t> ref_count_;
  std::queue<T> avaliable_;
};


template <typename T>
class Tetris {
 public:
  Tetris() = default;
  ~Tetris() = default;

  void Add(std::shared_ptr<TetrisColumn<T>> child);
  void Push(int32_t col_id, T element, int64_t element_id);
  bool Ready() const;
  std::unordered_map<int32_t, std::vector<std::pair<T, int64_t>>> Pop();

 private:
  std::unordered_map<int32_t, std::shared_ptr<TetrisColumn<T>>> columns_;
};

}  // namespace oneflow
#endif  // ONEFLOW_TASK_FSM_TETRIS_H_
