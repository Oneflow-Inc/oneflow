builtin.module {
  func.func @Graph_0(%arg0: tensor<1000x2048xf32>) -> tensor<2048x1000xf32> {
    %1 = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %2 = "tosa.transpose"(%arg0, %1) : (tensor<1000x2048xf32>, tensor<2xi32>) -> tensor<2048x1000xf32>
    func.return %2 : tensor<2048x1000xf32>
  }
}
