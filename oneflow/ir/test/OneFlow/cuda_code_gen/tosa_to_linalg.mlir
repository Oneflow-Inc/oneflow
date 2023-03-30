// RUN: oneflow-opt %s -ofjob-to-func --tosa-make-broadcastable -pass-pipeline="oneflow.job(tosa-to-linalg)" -func-to-ofjob

oneflow.job @GraphToRun_1(%arg0: tensor<2x5xi64>, %arg1: tensor<1xf32>) -> tensor<2x5xf32> {
    %2 = "tosa.cast"(%arg0) : (tensor<2x5xi64>) -> tensor<2x5xf32>
    %3 = "tosa.mul"(%2, %arg1) {shift = 0 : i32} : (tensor<2x5xf32>, tensor<1xf32>) -> tensor<2x5xf32>
    oneflow.return %3 : tensor<2x5xf32>
}