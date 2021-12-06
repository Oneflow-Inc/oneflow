# JIT in OneFlow

## Summary

There will be two modes of JIT in OneFlow:

- `oneflow.jit.trace`: `torch.jit.trace` behavior, create a JIT module from a vanilla `nn.module`
- `oneflow.jit.exec`: "Lazy Tensor" behavior

## Minimum Examples

- ### `oneflow.jit.trace`

  ```python
  class MyModule(oneflow.nn.Module):
      def __init__(self, N, M):
          super(MyModule, self).__init__()
          self.weight = oneflow.nn.Parameter(oneflow.rand(N, M))
          self.linear = oneflow.nn.Linear(N, M)

      def forward(self, input):  # python forward will not be run after the tracing
          output = self.linear(input)
          print(output) # will print Lazy tensor with no data, in Pytorch it prints tensor with data
          return output


  linear = oneflow.jit.trace(MyModule(2, 3))
  print(linear(oneflow.randn(2, 2))) # will print eager tensor with actual data
  ```

- ### `oneflow.jit.exec` decorator

  ```python
  @oneflow.jit.exec
  def any_function(a, b):
      return a + b

  z = any_function(x, y) # z is a lazy tensor
  print(z) # z is evaluated

  class MyModule(oneflow.nn.Module):
      def __init__(self, N, M):
          super(MyModule, self).__init__()
          self.weight = oneflow.nn.Parameter(oneflow.rand(N, M))
          self.linear = oneflow.nn.Linear(N, M)

      @oneflow.jit.exec
      def forward(self, input): # python forward will be run every time the module it is called
          output = self.linear(input)
          print(output) # will print eager tensor with data
          return output


  linear = MyModule(2, 3)
  print(linear(oneflow.randn(2, 2))) # will print eager tensor with actual data
  ```

- ### `oneflow.jit.exec` global mode

  ```python
  oneflow.jit.exec()
  a = oneflow.randn(2, 2)
  b = oneflow.randn(2, 2)
  c = a + b # no evaluation
  d = c + a + b # no evaluation
  print(c) # evaluation starts here
  ```

## Internal Details

There are mainly three components in the JIT system:

- JIT Interpreter: a special interpreter works on eager inputs and lazy intermediate tensors.
- Importer: convert OneFlow's representation to MLIR and and vice versa.
- Executor: three types of executor under development or consideration
  - Re-dispatch: convert every MLIR op to one User Op and have eager interpreter dispatch them. 10% performance boost over pure eager mode is expected. This will be used for `oneflow.jit.trace` mainly.
  - Batched op kernel: convert all MLIR ops to one UserOp and Kernel. This will be used to support CUDA graph.
  - Direct kernel launch: generate and launch kernel directly.

### Principle

- Avoid binding a OneFlow tensor with a MLIR Tensor Value. The side effect on OneFlow Tensor should be modeled with MLIR Op.
- Should not introduce any scope API in Python/Pybind11.
  - Should allow Python to abort the trace. Introducing scope will prevent this.
  - JIT Interpreter will check if it is in the right state when certain API is called.

### ~~Sequence of operations~~

(Not going to implement it because MLIR bash OpExpr is going to land)

```cpp
struct Op {
op_expr_seq: OpExpr
tensor_types: List<DataType>
inputs_in_the_seq: List<int_64_t>
}
using SeqOfOp = List<Op>
using PyUses = List<bool> // size of py_uses is the sum of tensor_types' sizes
```

For a typical "lazy tensor" implementation, there is usually a "sequence of operations" representation. At this point, we use Op Expr to form the sequence. Every time a new Op Expr is applied in JIT interpreter, we insert new element into the sequence and combine its hash with the existing sequence's. When an evaluation is triggered, we combine the hash of the collected sequence and `PyUses` to look up the cached MLIR.

### ~~Tracking Python reference of an Tensor~~

(This has been removed. Will do more research to see if it is needed.)

In exec mode, it is necessary to track if a lazy tensor is being referenced by Python and later it will be replaced with Eager Tensor.

- Use weak ptr to check if a lazy tensor is being referenced by Python.
- Edge cases

  - Intermediate tensor optimized away gets used after evaluation
    ```python3
    x = SomeOp(..)
    y = x + 1
    z = y + 2
    print(z) // evaluation, # y is folded
    print(y) // y, will be evaluated "again"
    ```

### Multi level cache

There are four level of cache

- Python forward function -> MLIR (trace mode only)
- Op Expr -> MLIR (trace mode and exec mode)
- Vanilla MLIR -> Optimized MLIR (trace mode and exec mode)
- MLIR Op -> Op Expression (only for re-dispatch execution)

### Ownership

- In exec mode, there is only one global MLIR Module which JIT Interpreter owns
- In trace mode, each module has its own MLIR Module
