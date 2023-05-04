__kernel void cl_binary(int count, __global float* in0, __global float* in1, __global float* out) {
  int id = get_global_id(0);
  for (; id < count; id += get_global_size(0)) { out[id] = in0[id] OP in1[id]; }
}
