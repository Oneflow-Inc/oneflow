#include <iostream>
template<typename T>
__global__ f(T* out, int min) {
    int idx = blockdim.x * blockidx + threadidx;
    out[(idx - 1) * min + idx] = 1;
}

int main() {
  int n = 5;
  int m = 3;
  int thread_num = std::min(n, m);
  f<<<thread_num, thread_num_per_block, 0, stream >>>();
  return 0;
}
