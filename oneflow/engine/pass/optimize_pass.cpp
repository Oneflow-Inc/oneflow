#include <unordered_map>
#include "glog/logging.h"
#include "oneflow/engine/pass/optimize_pass.h"

namespace oneflow {
namespace mola {

typedef std::unordered_map<std::string, OptimizePass::PassFactory>
    PassFactoryMap;

static PassFactoryMap *GlobalPassFactory() {
  static PassFactoryMap *global_factories = new PassFactoryMap;
  return global_factories;
}

void OptimizePass::Register(const std::string &pass_type,
                                      OptimizePass::PassFactory factory) {
  PassFactoryMap *factories = GlobalPassFactory();
  if (factories->count(pass_type) > 0) {
    DLOG(INFO) << "Pass (" << pass_type << ") has been registered more than once";
  }
  factories->emplace(pass_type, factory);
}

OptimizePass *OptimizePass::Create(const std::string &pass_type,
                                         const OptimizeOptions &options) {
  PassFactoryMap *factories = GlobalPassFactory();
  CHECK_GT(factories->count(pass_type), 0) << "Pass (" << pass_type
                                          << ") has not been registered";
  return (*factories)[pass_type](options);
}

void RunOptimizePass(const std::string &pass, OptimizeOptions &options) {
  auto optimize_pass = std::shared_ptr<mola::OptimizePass>(
      mola::OptimizePass::Create(pass, options));
  optimize_pass->Run();
}

}  // namespace mola
}  // namespace oneflow
