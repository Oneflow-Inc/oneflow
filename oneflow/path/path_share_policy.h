#ifndef _PATH_PATH_SHARE_POLICY_H_
#define _PATH_PATH_SHARE_POLICY_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "dag/register_info.h"
#include "path/path_type.h"
#include "layers/base_layer.h"

// PathSahrePolicy defines the information shared between path (e.g. register).
// During path building, each path needs to explicitly declare the sharing
// details. Currently assume the task_name is unique across all paths. This can
// be used in memory allocation, layer_blob_to_register_blob implementation.
namespace oneflow {

// About placeholder
// In kModelUpdatePath -> (kModel) -> kDataPath: neither producer or consumer is
// a placeholder.
// In kDataPath -> (kModelDiff) -> kModelUpdatePath, sometimes, the consumer is
// a placeholder.
// In kModelUpdatePath -> (kModel) -> kModelStorePath, the consumer is a 
// placeholder.
// In kModelLoadPath -> (kData) -> kModelUpdatePath, the producer is a 
// placeholder

enum class PathSharingRole {
  kProducer = 0,
  kConsumer
};

enum class TaskDirection {
  kForward = 0,
  kBackward
};

enum class TaskPlaceholder {
  kYes = 0,
  kNo
};

enum class RegisterOwner {
  kYes = 0,
  kNo
};

struct PathSharingDetail {
  PathSharingRole role;
  PathType path_type;
  std::string net_name;
  std::string segment_name;
  RegisterType register_type;
  TaskDirection task_direction;
  TaskPlaceholder task_placeholder;
  RegisterOwner register_owner;
};

struct PathSharingDescriptor {
  PathSharingDetail producer_detail;
  PathSharingDetail consumer_detail;

  PathSharingDescriptor(const PathSharingDescriptor& other) = default;
  PathSharingDescriptor& operator=(const PathSharingDescriptor& other) = default;
};

}  // namespace oneflow
#endif  // _PATH_PATH_SHARE_POLICY_H_
