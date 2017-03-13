#ifndef _PATH_BASE_PATH_H_
#define _PATH_BASE_PATH_H_
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include "common/common.h"
#include "dag/segment_task_map.h"
#include "dag/dag_builder.h"
#include "path/path_type.h"
#include "path/path_share_policy.h"

namespace caffe {

template <typename Dtype>
class PathManager;

template <typename Dtype>
class TaskDag;

template <typename Dtype>
class BasePath {
public:
  BasePath(PathType path_type, PathManager<Dtype>* path_manager)
    : path_type_(path_type), path_manager_(path_manager) {
  }
  virtual ~BasePath() {}

  BasePath(const BasePath& other) = delete;
  BasePath& operator=(const BasePath& other) = delete;

  virtual void Build() = 0;
  virtual void Setup() = 0;

  std::shared_ptr<DagBuilder<Dtype>> GetDagBuilder(
    const std::string& net_name) const;
  PathType path_type() const { return path_type_; }

  std::vector<int32_t> GetDeviceIDs(
    const std::string& net_name, const std::string& segment_name) const;
  std::shared_ptr<TaskDag<Dtype>> GetCrossPathTaskDag(
    const PathSharingDetail& sharing_detail, int32_t device_id) const;

protected:
  const PathType path_type_;
  PathManager<Dtype>* path_manager_;
  // There may be multiple networks in a path. Each network is created and setup
  // by a particular DagBuiler object. |dag_builders_| maintains a map between
  // the network name and the corresponding DagBuilder object.
  std::unordered_map<std::string, std::shared_ptr<DagBuilder<Dtype>>>
    dag_builder_dict_;
};

template <typename Dtype>
std::shared_ptr<DagBuilder<Dtype>> BasePath<Dtype>::GetDagBuilder(
  const std::string& net_name) const {
  auto dag_builder_it = dag_builder_dict_.find(net_name);
  CHECK(dag_builder_it != dag_builder_dict_.end());
  return dag_builder_it->second;
}

template <typename Dtype>
std::vector<int32_t> BasePath<Dtype>::GetDeviceIDs(
  const std::string& net_name, const std::string& segment_name) const {
  auto dag_builder = this->GetDagBuilder(net_name);
  return dag_builder->GetDeviceIDs(segment_name);
}

template <typename Dtype>
std::shared_ptr<TaskDag<Dtype>> BasePath<Dtype>::GetCrossPathTaskDag(
  const PathSharingDetail& sharing_detail,
  int32_t device_id) const {

  auto dag_builder = this->GetDagBuilder(sharing_detail.net_name);
  return dag_builder->GetCrossPathTaskDag(sharing_detail, device_id);
}

}  // namespace caffe
#endif  // _PATH_BASE_PATH_H_
