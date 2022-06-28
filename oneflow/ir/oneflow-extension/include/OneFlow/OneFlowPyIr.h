#include <glog/logging.h>
#include <functional>
#include <unordered_map>
#include <utility>

class ast {};

class LR_JIT final {
 public:
  void Register(const std::string& function_id, std::function<double(double, int64_t)> lr_func);
  bool Invoke(const std::string& function_id, double& lr, double base_lr, int64_t step);

 private:
  std::unordered_map<std::string, std::function<double(double, int64_t)>> function_id2lr_func_;
};

static std::function<double(double, int64_t)> AstToJIT(const ast& ast,
                                                       const std::string& function_id);
