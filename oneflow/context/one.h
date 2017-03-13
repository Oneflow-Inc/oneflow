#ifndef _CONTEXT_ONE_H_
#define _CONTEXT_ONE_H_
#include <memory>
#include <string>
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
class ConfigParser;
class IDMap;
template <typename Dtype>
class CommBus;

template <typename Dtype>
class JobManager;

template <typename Dtype>
class NodeManager;

template <typename Dtype>
class PathManager;

template <typename Dtype>
class TheOne {
 public:
  ~TheOne();
  static TheOne<Dtype>& Get();
  static void InitResource(const std::string& solver_name);
  static void InitJob(const SolverProto& param);
  static void InitJob2(const SolverProto& param);
  static void InitThread();
  static void InitNetwork();
  static void FinalizeThread();
  static std::shared_ptr<ConfigParser> config_parser();
  static std::shared_ptr<ConfigParser> ps_config_parser();
  static std::shared_ptr<IDMap> id_map();
  static std::shared_ptr<CommBus<Dtype>> comm_bus();
  static std::shared_ptr<NodeManager<Dtype>> node_manager();
  static std::shared_ptr<JobManager<Dtype>> job_manager();
  static std::shared_ptr<PathManager<Dtype>> path_manager();

 private:
  static std::unique_ptr<TheOne<Dtype>> singleton_;
  static std::shared_ptr<ConfigParser> config_parser_;
  static std::shared_ptr<ConfigParser> ps_config_parser_;
  static std::shared_ptr<IDMap> id_map_;

  static std::shared_ptr<CommBus<Dtype>> comm_bus_;
  static std::shared_ptr<JobManager<Dtype>> job_manager_;
  static std::shared_ptr<NodeManager<Dtype>> node_manager_;

  static std::shared_ptr<PathManager<Dtype>> path_manager_;

  TheOne();
  TheOne(const TheOne& other) = delete;
  TheOne& operator=(const TheOne& other) = delete;
};

template <typename Dtype>
inline TheOne<Dtype>& TheOne<Dtype>::Get() {
  if (!singleton_.get()) {
    singleton_.reset(new TheOne<Dtype>());
  }
  return *singleton_;
}

template <typename Dtype>
inline std::shared_ptr<ConfigParser> TheOne<Dtype>::config_parser() {
  return Get().config_parser_;
}

template <typename Dtype>
inline std::shared_ptr<ConfigParser> TheOne<Dtype>::ps_config_parser() {
  return Get().ps_config_parser_;
}

template <typename Dtype>
inline std::shared_ptr<IDMap> TheOne<Dtype>::id_map() {
  return Get().id_map_;
}

template <typename Dtype>
inline std::shared_ptr<CommBus<Dtype>> TheOne<Dtype>::comm_bus() {
  return Get().comm_bus_;
}

template <typename Dtype>
inline std::shared_ptr<NodeManager<Dtype>> TheOne<Dtype>::node_manager() {
  return Get().node_manager_;
}

template <typename Dtype>
inline std::shared_ptr<JobManager<Dtype>> TheOne<Dtype>::job_manager() {
  return Get().job_manager_;
}

template <typename Dtype>
inline std::shared_ptr<PathManager<Dtype>> TheOne<Dtype>::path_manager() {
  return Get().path_manager_;
}

}  // namespace caffe
#endif  // _CONTEXT_ONE_H_
