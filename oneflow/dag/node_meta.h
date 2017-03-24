#ifndef _DAG_NODE_META_H_
#define _DAG_NODE_META_H_
#include <string>
#include <vector>
#include "common/shape.h"
#include "common/task_type.h"
#include "context/placement_info.h"

namespace oneflow {
// Hold the name and the shape of a data blob
class BlobMeta {
public:
  BlobMeta(){}
  explicit BlobMeta(const std::string& name) : name_(name) {}
  const Shape& shape() const { return shape_; }
  Shape& mutable_shape() { return shape_; }
  const std::string& name() const { return name_; }
  std::string& mutable_name() { return name_; }
private:
  std::string name_;
  Shape shape_;
};

template <typename Dtype>
class BaseLayer;
template <typename Dtype>
class LayerMeta {
 public:
  explicit LayerMeta(const std::string& type) : type_(type) { }
  const std::string& type() const { return type_; }
  std::string& mutable_type() { return type_; }
  const std::string& param_str() const { return param_str_; }
  std::string& mutable_param_str() { return param_str_; }
  std::shared_ptr<const BaseLayer<Dtype>> layer() const { return layer_; }
  std::shared_ptr<BaseLayer<Dtype>>& mutable_layer() { return layer_; }
  const PlacementInfo& placement_info() const { return placement_info_; }
  PlacementInfo& mutable_placement_info() { return placement_info_; }
  bool has_BP() const { return has_BP_; }
  bool& mutable_has_BP() { return has_BP_; }
 private:
  std::string type_;
  std::string param_str_;
  std::shared_ptr<BaseLayer<Dtype>> layer_;
  PlacementInfo placement_info_;
  bool has_BP_{ false };
};

class PlacementGroupMeta {
public:
  PlacementGroupMeta() {}

  const PlacementInfo& placement_info() const { return placement_info_; }
  PlacementInfo& mutable_placement_info() { return placement_info_; }
private:
  PlacementInfo placement_info_;
};

class ClusteringMeta {
public:
  ClusteringMeta() {}

  const std::vector<std::string>& layer_names() const { return layer_names_; }
  std::vector<std::string>& mutable_layer_names() { return layer_names_; }
private:
  std::vector<std::string> layer_names_;
};

class EnvelopeMeta {
public:
  EnvelopeMeta() {}

  const std::vector<std::string>& blob_names() const { return blob_names_; }
  std::vector<std::string>& mutable_blob_names() { return blob_names_; }
private:
  std::vector<std::string> blob_names_;
};

class SegmentMeta {
public:
  SegmentMeta() {}

  const std::vector<std::string>& layer_names() const { return layer_names_; }
  std::vector<std::string>& mutable_layer_names() { return layer_names_; }
  const PlacementInfo& placement_info() const { return placement_info_; }
  PlacementInfo& mutable_placement_info() { return placement_info_; }
  bool has_BP() const { return has_BP_; }
  bool& mutable_has_BP() { return has_BP_; }
private:
  std::vector<std::string> layer_names_;
  PlacementInfo placement_info_;
  bool has_BP_{ false };
};

class StageMeta {
public:
  StageMeta() {}

  const int32_t machine_id() const { return machine_id_; }
  int32_t& mutable_machine_id() { return machine_id_; }
  const std::string segment_name() const { return segment_name_; }
  std::string& mutable_segment_name() { return segment_name_; }
  bool has_BP() const { return has_BP_; }
  bool& mutable_has_BP() { return has_BP_; }
private:
  int32_t machine_id_;
  std::string segment_name_;
  bool has_BP_;
};

class PipeMeta {
  public:
    explicit PipeMeta() {}

    int32_t thread_id() const { return thread_id_; }
    int32_t& mutable_thread_id() { return thread_id_; }
    TaskType task_type() const { return task_type_; }
    TaskType& mutable_task_type() { return task_type_; }
    bool has_BP() const { return has_BP_; }
    bool& mutable_has_BP() { return has_BP_; }
  private:
    int32_t thread_id_;
    TaskType task_type_{ TaskType::kUnknownTask };
    bool has_BP_{ false };
};

class ActorMeta {
public:
  ActorMeta() {}
  int32_t task_id() const { return task_id_; }
  int32_t& mutable_task_id() { return task_id_; }
  TaskType task_type() const { return task_type_; }
  TaskType& mutable_task_type() { return task_type_; }
  const std::string& pipe_name() const { return pipe_name_; }
  std::string& mutable_pipe_name() { return pipe_name_; }
  bool is_forward() const { return is_forward_; }
  bool& mutable_is_forward() { return is_forward_; }
private:
  int32_t task_id_;
  TaskType task_type_{ TaskType::kUnknownTask };
  std::string pipe_name_;
  bool is_forward_;
};

class EventMeta {
 public:
  EventMeta() {}

  void SetForwardReceiver(int32_t forward_receiver_id) {
    forward_receiver_id_ = forward_receiver_id;
  }
  void SetBackwardReceiver(int32_t backward_receiver_id) {
    backward_receiver_id_ = backward_receiver_id;
  }
  const int32_t forward_receiver_id() const { return forward_receiver_id_; }
  const int32_t backward_receiver_id() const { return backward_receiver_id_; }
  int32_t thread_id() const { return thread_id_; }
  int32_t& mutable_thread_id() { return thread_id_; }

 private:
   int32_t thread_id_{ -2 };
   int32_t forward_receiver_id_{ -1 };
   int32_t backward_receiver_id_{ -1 };
};
}  // namespace oneflow
#endif  // _DAG_NODE_META_H_
