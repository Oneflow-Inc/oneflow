#include <unordered_map>
#include "glog/logging.h"
#include "oneflow/core/compiler/of2xla/pass/xla_optimize_pass.h"

namespace oneflow {
namespace mola {

typedef std::unordered_map<std::string, XlaOptimizePass::PassFactory>
    PassFactoryMap;

static PassFactoryMap *GlobalPassFactory() {
  PassFactoryMap *global_factories = new PassFactoryMap;
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

XlaOptimizePass *Create(const std::string &pass_type,
                               const OptimizeOptions &options) {
  PassFactoryMap *factories = GlobalPassFactory();
  CHECK_GT(factories->count(pass_type), 0) << "Pass (" << pass_type
                                          << ") has not been registered";
  return (*factories)[pass_type](options);
}

}  // namespace mola
}  // namespace oneflow
