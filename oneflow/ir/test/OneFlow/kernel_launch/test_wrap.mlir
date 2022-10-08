// oneflow-opt %s
"func.func"() ({
^bb0(%arg0: tensor<2xf32>):
  %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
  "func.return"(%0) : (tensor<2xf32>) -> ()
}) {compiled = "true", function_type = (tensor<2xf32>) -> tensor<2xf32>, sym_name = "relu2D0"} : () -> ()
