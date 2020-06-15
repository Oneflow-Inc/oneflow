import oneflow as flow

# test file names and methods names are starts with `test'


def test_foo(test_case):
    # only one arg required
    # you can use `test_case' like unittest.TestCase instance
    pass


@flow.unittest.num_nodes_required(2)
def test_bar(test_case):
    # default num_nodes_required is 1
    pass
