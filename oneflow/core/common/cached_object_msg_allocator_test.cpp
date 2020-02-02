#include "oneflow/core/common/cached_object_msg_allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

TEST(CachedObjectMsgAllocator, constructor_destructor) {
  CachedObjectMsgAllocator allocator(20, 100);
}

TEST(CachedObjectMsgAllocator, Allocator) {
  CachedObjectMsgAllocator allocator(20, 100);
  char* mem_ptr = allocator.Allocate(1024);
  allocator.Deallocate(mem_ptr, 1024);
}

class TestObjMsgAllocator final : public ObjectMsgAllocator {
 public:
  TestObjMsgAllocator(int* cnt) : ObjectMsgAllocator(), cnt_(cnt) {}

  char* Allocate(std::size_t size) override {
    ++*cnt_;
    return ObjectMsgDefaultAllocator::GlobalObjectMsgAllocator()->Allocate(size);
  }
  void Deallocate(char* ptr, std::size_t size) override {
    --*cnt_;
    return ObjectMsgDefaultAllocator::GlobalObjectMsgAllocator()->Deallocate(ptr, size);
  }

 private:
  int* cnt_;
};

TEST(CachedObjectMsgAllocator, no_memory_leak) {
  int cnt = 0;
  {
    TestObjMsgAllocator backend_allocator(&cnt);
    CachedObjectMsgAllocator allocator(&backend_allocator, 20, 100);
    for (int i = 0; i < 100; ++i) {
      char* mem_ptr = allocator.Allocate(1024);
      allocator.Deallocate(mem_ptr, 1024);
    }
    char* mem_ptr[100];
    for (int i = 0; i < 100; ++i) { mem_ptr[i] = allocator.Allocate(1024); }
    for (int i = 0; i < 100; ++i) { allocator.Deallocate(mem_ptr[i], 1024); }
  }
  ASSERT_EQ(cnt, 0);
}

TEST(CachedObjectMsgAllocator, stacked_object_msg_allocator) {
  int cnt = 0;
  {
    TestObjMsgAllocator backend_allocator(&cnt);
    CachedObjectMsgAllocator backend_cache_allocator(&backend_allocator, 21, 100);
    CachedObjectMsgAllocator allocator(&backend_cache_allocator, 20, 100);
    for (int i = 0; i < 100; ++i) {
      char* mem_ptr = allocator.Allocate(1024);
      allocator.Deallocate(mem_ptr, 1024);
    }
    char* mem_ptr[100];
    for (int i = 0; i < 100; ++i) { mem_ptr[i] = allocator.Allocate(1024); }
    for (int i = 0; i < 100; ++i) { allocator.Deallocate(mem_ptr[i], 1024); }
  }
  ASSERT_EQ(cnt, 0);
}

TEST(ThreadUnsafeObjectMsgAllocator, stacked_object_msg_allocator) {
  int cnt = 0;
  {
    TestObjMsgAllocator backend_allocator(&cnt);
    ThreadUnsafeObjectMsgAllocator backend_cache_allocator(&backend_allocator, 21, 100);
    ThreadUnsafeObjectMsgAllocator allocator(&backend_cache_allocator, 20, 100);
    for (int i = 0; i < 100; ++i) {
      char* mem_ptr = allocator.Allocate(1024);
      allocator.Deallocate(mem_ptr, 1024);
    }
    char* mem_ptr[100];
    for (int i = 0; i < 100; ++i) { mem_ptr[i] = allocator.Allocate(1024); }
    for (int i = 0; i < 100; ++i) { allocator.Deallocate(mem_ptr[i], 1024); }
  }
  ASSERT_EQ(cnt, 0);
}

}  // namespace test

}  // namespace oneflow
