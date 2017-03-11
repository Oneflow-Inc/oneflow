#include "common/rng.h"

namespace caffe{
//declaration of static variable
std::shared_ptr<RNG> RNG::sigleton_;

//RandomGeneraotr::Generator
RNG::Generator::Generator() :
  seed_(0), offset_(0), rng_type_(CURAND_RNG_PSEUDO_DEFAULT),
  ordering_(CURAND_ORDERING_PSEUDO_BEST){
  CURAND_CHECK(curandCreateGeneratorHost(&generator_, rng_type_));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed_));
  CURAND_CHECK(curandSetGeneratorOffset(generator_, offset_));
  CURAND_CHECK(curandSetGeneratorOrdering(generator_, ordering_));
}
RNG::Generator::Generator(const size_t seed) :
  seed_(seed), offset_(0), rng_type_(CURAND_RNG_PSEUDO_DEFAULT),
  ordering_(CURAND_ORDERING_PSEUDO_BEST){
  CURAND_CHECK(curandCreateGeneratorHost(&generator_, rng_type_));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed_));
  CURAND_CHECK(curandSetGeneratorOffset(generator_, offset_));
  CURAND_CHECK(curandSetGeneratorOrdering(generator_, ordering_));
}
RNG::Generator::Generator(const size_t seed, const size_t offset,
  const curandRngType_t rng_type, const curandOrdering_t ordering) :
  seed_(seed), offset_(offset), rng_type_(rng_type), ordering_(ordering){
  CURAND_CHECK(curandCreateGeneratorHost(&generator_, rng_type_));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed_));
  CURAND_CHECK(curandSetGeneratorOffset(generator_, offset_));
  CURAND_CHECK(curandSetGeneratorOrdering(generator_, ordering_));
}
RNG::Generator::~Generator(){
  CURAND_CHECK(curandDestroyGenerator(generator_));
}
void RNG::Generator::set_seed(const size_t seed){
  seed_ = seed;
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed_));
}
void RNG::Generator::set_offset(const size_t offset){
  offset_ = offset;
  CURAND_CHECK(curandSetGeneratorOffset(generator_, offset));
}
void RNG::Generator::set_ordering(const curandOrdering_t ordering){
  ordering_ = ordering;
  CURAND_CHECK(curandSetGeneratorOrdering(generator_, ordering_));
}
}//namespace caffe