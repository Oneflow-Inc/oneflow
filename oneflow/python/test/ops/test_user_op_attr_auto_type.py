import oneflow as flow

def _test_user_op_attr_auto_type(attr1, attr2):
    return (
        flow.user_op_builder("test")
        .Op("test_user_op_attr_auto_type")
        .Input("in")
        .Output("out")
        .Attr("int1", attr1)
        .Attr("int2", attr2, "AttrTypeInt32")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

def test_user_op_attr_auto_type(test_case):
    _test_user_op_attr_auto_type(1, 2)
