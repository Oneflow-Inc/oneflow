#ifndef ONEFLOW_CORE_REGISTER_LOD_VIEW_H_
#define ONEFLOW_CORE_REGISTER_LOD_VIEW_H_

#include "oneflow/core/register/pod_ptr.h"
#include "oneflow/core/register/lod_tree.pb.h"

namespace oneflow {

class LoDViewBase {
 protected:
  typedef std::vector<std::vector<int64_t>> LoDVec;

  LoDViewBase(PodPtr lod_ptr, int64_t num_of_lod_levels);
  LoDViewBase(const LoDViewBase& rhs) = default;
  ~LoDViewBase() = default;

  LoDVec InitOffsetVecFromPtr() const;
  void FlushOffsetVecToPtr(const LoDVec& offset_lod_vec);

  LoDVec GetLengthLoDVecFromOffsetLoDVec(const LoDVec& offset_lod_vec) const;
  LoDVec GetOffsetLoDVecFromLengthLoDVec(const LoDVec& length_lod_vec) const;

  int64_t* ptr_;
  int64_t num_of_lod_levels_;
  int64_t max_reserved_size_for_lod_;
};

class OffsetLoDView final : public LoDViewBase {
 public:
  OffsetLoDView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels), offset_lod_vec_() {}
  OffsetLoDView(const OffsetLoDView& rhs)
      : LoDViewBase(rhs), offset_lod_vec_(rhs.offset_lod_vec_) {}

  int64_t GetOffset(size_t level, size_t pos);

 private:
  LoDVec offset_lod_vec_;
};

class OffsetLoDMutView final : public LoDViewBase {
 public:
  OffsetLoDMutView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {}
  OffsetLoDMutView(const OffsetLoDMutView& rhs) : LoDViewBase(rhs) {}

  void SetOffset(const LoDVec& offset_lod_vec);
};

class LengthLoDView final : public LoDViewBase {
 public:
  LengthLoDView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels), length_lod_vec_() {}
  LengthLoDView(const LengthLoDView& rhs) : LoDViewBase(rhs) {}

  int64_t GetLength(size_t level, size_t pos);

 private:
  LoDVec length_lod_vec_;
};

class LengthLoDMutView final : public LoDViewBase {
 public:
  LengthLoDMutView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {}
  LengthLoDMutView(const LengthLoDMutView& rhs) : LoDViewBase(rhs) {}

  void SetLength(const LoDVec& length_lod_vec);
};

struct TreeLoDHelper final {
  static void FindLevelMutNodes(int64_t expected_level, LoDTree* lod_tree,
                                std::vector<LoDTree*>* leaves);
  static void FindLevelNodes(int64_t expected_level, const LoDTree* lod_tree,
                             std::vector<const LoDTree*>* leaves);
  static void UpdateInnerNode(LoDTree* lod_tree);
};

class TreeLoDView final : public LoDViewBase {
 public:
  TreeLoDView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {
    Init();
  }
  TreeLoDView(const TreeLoDView& rhs) = default;

  LoDTree lod_tree() const { return lod_tree_; }

 private:
  void Init();
  void InitTree();
  LoDTree lod_tree_;
};

class TreeLoDMutView final : public LoDViewBase {
 public:
  TreeLoDMutView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {}
  TreeLoDMutView(const TreeLoDMutView& rhs) = default;

  void UpdateLoD(const LoDTree& lod_tree) const;
};

class CoordinateLoDMutView final : public LoDViewBase {
 public:
  CoordinateLoDMutView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels), lod_ptr_(lod_ptr) {}
  CoordinateLoDMutView(const CoordinateLoDMutView& rhs) = default;

  using Coordinate = std::vector<int64_t>;
  using OffsetLength = std::pair<int64_t, int64_t>;
  using Coordinate2OffsetLength = std::map<Coordinate, OffsetLength>;
  void UpdateLoD(const Coordinate2OffsetLength& coord2offset_length) const;

 private:
  void MakeLoDTree(const Coordinate2OffsetLength& coord2offset_length, LoDTree*) const;

  PodPtr lod_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_LOD_VIEW_H_
