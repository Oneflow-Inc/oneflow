#ifndef RNG_H_
#define RNG_H_

#include <curand.h>
#include <glog/logging.h>

#include <memory>

#include "device/device_alternate.h"
namespace caffe {
class RNG {
 public:
  class Generator {
   public:
    Generator();
    explicit Generator(const size_t seed);
    explicit Generator(const size_t seed, const size_t offset,
      const curandRngType_t rng_type, const curandOrdering_t ordering);
    ~Generator();

    inline size_t get_seed() const { return seed_; }
    inline size_t get_offset() const { return offset_; }
    inline curandRngType_t get_rng_type_() const { return rng_type_; }
    inline curandOrdering_t get_ordering() const { return ordering_; }
    inline curandGenerator_t get_generator() { return generator_; }

    void set_seed(const size_t seed);
    void set_offset(const size_t offset);
    void set_ordering(const curandOrdering_t ordering);

   private:
    size_t seed_;
    size_t offset_;
    curandRngType_t rng_type_;
    curandOrdering_t ordering_;
    curandGenerator_t generator_;

    Generator(const Generator& other) = delete;
    Generator& operator=(const Generator& other) = delete;
  };

  ~RNG() = default;

  inline static void set_seed(const size_t seed) {
    get_generator().set_seed(seed);
  }
  //  NOTE(xcdu):2015.11.5 Just for test;
  inline static size_t get_seed() {
    return get_generator().get_seed();
  }
  inline static void set_offset(const size_t offset) {
    get_generator().set_offset(offset);
  }
  inline static void set_ordering(const curandOrdering_t ordering) {
    get_generator().set_ordering(ordering);
  }
  inline static curandGenerator_t generator() {
    return get_generator().get_generator();
  }

 private:
  inline static RNG& get() {
    if (!sigleton_.get()) {
      sigleton_.reset(new RNG());
    }
    return *sigleton_;
  }
  inline static Generator& get_generator() {
    if (!get().generator_.get()) {
      get().generator_.reset(new Generator());
    }
    return *(get().generator_);
  }

  std::shared_ptr<Generator> generator_;
  static std::shared_ptr<RNG> sigleton_;
};
}  // namespace caffe
#endif  // RNG_H_
