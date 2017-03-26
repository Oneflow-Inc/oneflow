#ifndef _PATH_PATH_MANAGER_H_
#define _PATH_PATH_MANAGER_H_
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include "path/base_path.h"
#include "proto/oneflow.pb.h"
#include "proto/proto_io.h"

namespace std {
  template<>
  struct hash<oneflow::PathType>{
    typedef oneflow::PathType argument_type;
    typedef size_t result_type;
    result_type operator () (const argument_type& x) const{
      using type = typename std::underlying_type<argument_type>::type;
      return std::hash<type>()(static_cast<type>(x));
    }
  };
}

namespace oneflow {
template <typename Dtype>
class PathManager {
public:
  PathManager() = default;
  ~PathManager() = default;

  PathManager(const PathManager& other) = delete;
  PathManager& operator=(const PathManager& other) = delete;

  void Initialize(const SolverProto&param);

  // For cross-path dependency, the producer path is created by the creation of
  // the consumer path. The dependency can only be declared by the consumer 
  // since the producer is not aware of the dependency at the time it is created.
  // The consumer declares the cross dependency by calling the following
  // |AddPathSharing| while it is created in |Path.Build| to notify PathManager,
  // the producer and the consumer to know the dependency. An example usage can
  // be found in |ModelUpdatePath.CreateModelUpdateDags|.
  void AddPathSharing(const PathSharingDescriptor& path_mapping_desc);

  std::shared_ptr<BasePath<Dtype>> GetPath(PathType type) const;
  // Some paths like kModelLoadPath and kModelStorePath need to know the update
  // segment name while establishing the cross-path sharing.
  // TODO(jiyuan): Current interface and implementation are just a temporal 
  // work-around.
  std::string GetUpdateSegmentNameInModelUpdatePath() const;
private:
  std::unordered_map<PathType, std::shared_ptr<BasePath<Dtype>>> path_dict_;
  std::vector<PathSharingDescriptor> sharing_descriptors_;

  void Build(const SolverProto& param);

  void CompleteOneRegisterInfo(const PathSharingDescriptor& sharing_descriptor);

  void Connect();
  void Setup();
};
}  // namespace oneflow
#endif  // _PATH_PATH_MANAGER_H_
