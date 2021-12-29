namespace oneflow{

namespace {

void SimplifyBroadcastDims(size_t num_a_dims, const int64_t* a_dims, size_t num_b_dims,
                            const int64_t* b_dims, size_t* simplified_num_dims,
                            int64_t* simplified_a_dims, int64_t* simplified_b_dims) {
  const size_t num_max_dims = std::max(num_a_dims, num_b_dims);
  auto MakeGetDim = [num_max_dims](size_t num_dims, const int64_t* dims) {
    const int64_t num_padding_dims = num_max_dims - num_dims;
    return [num_padding_dims, dims](size_t index) {
      return index < num_padding_dims ? 1 : dims[index - num_padding_dims];
    };
  };
  auto GetADim = MakeGetDim(num_a_dims, a_dims);
  auto GetBDim = MakeGetDim(num_b_dims, b_dims);
  *simplified_num_dims = 0;
  bool prev_broadcast_a = false;
  bool prev_broadcast_b = false;
  for (int64_t i = 0; i < num_max_dims; ++i) {
    const int64_t a_dim = GetADim(i);
    const int64_t b_dim = GetBDim(i);
    const int64_t broadcast_dim = std::max(a_dim, b_dim);
    CHECK_GT(broadcast_dim, 0);
    const bool broadcast_a = (a_dim == 1);
    const bool broadcast_b = (b_dim == 1);
    CHECK((a_dim == broadcast_dim) || broadcast_a);
    CHECK((b_dim == broadcast_dim) || broadcast_b);
    if (broadcast_dim == 1) {
      continue;
    } else if (*simplified_num_dims != 0
               && (prev_broadcast_a == broadcast_a && prev_broadcast_b == broadcast_b)) {
      simplified_a_dims[*simplified_num_dims - 1] *= a_dim;
      simplified_b_dims[*simplified_num_dims - 1] *= b_dim;
    } else {
      simplified_a_dims[*simplified_num_dims] = a_dim;
      simplified_b_dims[*simplified_num_dims] = b_dim;
      *simplified_num_dims += 1;
      prev_broadcast_a = broadcast_a;
      prev_broadcast_b = broadcast_b;
    }
  }

}

}
}