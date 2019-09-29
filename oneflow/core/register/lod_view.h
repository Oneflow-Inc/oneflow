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

struct TreeLodHelper final {
  static void FindLevelNodes(int64_t expected_level, LodTree* lod_tree,
                             std::vector<LodTree*>* leaves);
  static void UpdateInnerNode(LodTree* lod_tree);

 private:
  static void FindLevelNodes(int64_t expected_level, int64_t cur_level, LodTree* lod_tree,
                             std::vector<LodTree*>* leaves);
};

class TreeLodView final : public LoDViewBase {
 public:
  TreeLodView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {
    Init();
  }
  TreeLodView(const TreeLodView& rhs) = default;

  const LodTree& lod_tree() const { return lod_tree_; }

 private:
  void Init();
  void InitTree();
  LodTree lod_tree_;
};

class TreeLodMutView final : public LoDViewBase {
 public:
  TreeLodMutView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {}
  TreeLodMutView(const TreeLodMutView& rhs) = default;

  void set_lod_tree(LodTree&& lod_tree) const;
};

class CoordinateLodMutView final : public LoDViewBase {
 public:
  CoordinateLodMutView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {}
  CoordinateLodMutView(const CoordinateLodMutView& rhs) = default;

  using CoordOffsetLength = std::tuple<std::vector<int64_t>, int64_t, int64_t>;

  void set_lod_tree(const std::vector<CoordOffsetLength>& corrd_offset_lengths) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_LOD_VIEW_H_
