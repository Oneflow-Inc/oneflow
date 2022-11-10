namespace oneflow {
namespace dtr {
class Env {
 public:
  double time_now() { return time_now_; }
  void add_time(double time) { time_now_ += time; }

 private:
  double time_now_;
};
}  // namespace dtr
}  // namespace oneflow
