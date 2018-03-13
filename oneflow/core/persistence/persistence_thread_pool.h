#ifndef oqwd
#define oqwd

namespace oneflow {

class PersistenceThreadPool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistenceThreadPool);
  PersistenceThreadPool() = delete;
  ~PersistenceThreadPool() = default;

  OF_SINGLETON(PersistenceThreadPool);

  void Schedule(std::function<void()> fn) { TODO(); }

 private:
  PersistenceThreadPool(const Plan& plan) { TODO(); }
};

}  // namespace oneflow

#endif  // oqwd
