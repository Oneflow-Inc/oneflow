#include <unordered_map>
#include "glog/logging.h"
#include "oneflow/xla/of2xla/pass/xla_optimize_pass.h"

DEFINE_int32(clustering_minimum_nodes,
             EnvToInt(FLAGS_clustering_minimum_nodes, 1),
             "Minium nodes of a cluster after clustering.");
DEFINE_int32(clustering_maximum_nodes,
             EnvToInt(FLAGS_clustering_maximum_nodes, 1000),
             "Maxium nodes of a cluster after clustering.");
DEFINE_bool(strict_clustering, EnvToBool(FLAGS_strict_clustering, true),
            "Option to clustering with strict dependencies analysis.");

namespace oneflow {
namespace mola {

typedef std::unordered_map<std::string, XlaOptimizePass::PassFactory>
    PassFactoryMap;

static PassFactoryMap *GlobalPassFactory() {
  static PassFactoryMap *global_factories = new PassFactoryMap;
  return global_factories;
}

void XlaOptimizePass::Register(const std::string &pass_type,
                                      XlaOptimizePass::PassFactory factory) {
  PassFactoryMap *factories = GlobalPassFactory();
  if (factories->count(pass_type) > 0) {
    DLOG(INFO) << "Pass (" << pass_type << ") has been registered more than once";
  }
  factories->emplace(pass_type, factory);
}

XlaOptimizePass *XlaOptimizePass::Create(const std::string &pass_type,
                                         const OptimizeOptions &options) {
  PassFactoryMap *factories = GlobalPassFactory();
  CHECK_GT(factories->count(pass_type), 0) << "Pass (" << pass_type
                                          << ") has not been registered";
  return (*factories)[pass_type](options);
}

OptimizeOptions CreateDefaultOptimizeOptions() {
  OptimizeOptions options;
  options.clustering_minimum_nodes = FLAGS_clustering_minimum_nodes;
  options.clustering_maximum_nodes = FLAGS_clustering_maximum_nodes;
  options.strict_clustering = FLAGS_strict_clustering;
  return options;
}

void RunOptimizePass(const std::string &pass, OptimizeOptions &options) {
  auto optimize_pass = std::shared_ptr<mola::XlaOptimizePass>(
      mola::XlaOptimizePass::Create(pass, options));
  optimize_pass->Run();
}

}  // namespace mola
}  // namespace oneflow
