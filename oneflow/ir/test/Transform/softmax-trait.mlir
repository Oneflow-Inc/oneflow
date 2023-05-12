// RUN: oneflow-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
}
