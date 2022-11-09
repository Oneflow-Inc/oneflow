import unittest
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=1)
def _test_global_broadcast_tensors(
    test_case, input_shape, other_shape, placement, x_sbp, y_sbp
):
    x = random_tensor(len(input_shape), *input_shape).to_global(
        placement=placement, sbp=x_sbp
    )
    y = random_tensor(len(other_shape), *other_shape).to_global(
        placement=placement, sbp=y_sbp
    )
    return torch.broadcast_tensors(x, y)


class TestGlobalBroadcastOps(flow.unittest.TestCase):
    # flow.broadcast_shapes‘s input are shapes, so it can't be tested in global mode
    # flow.broadcast_to is an alias of flow.expand, so its global tests are same as flow.expand's

    @globaltest
    def test_global_tensors(test_case):
        shapes = [((2, 2), (2, 2, 2)), ((1, 2), (3, 1))]
        for input_shape, other_shape in shapes:
            for placement in all_placement():
                for x_sbp in all_sbp(
                    placement,
                    max_dim=2,
                    valid_split_axis=[x for x in input_shape if x != 1],
                ):
                    for y_sbp in all_sbp(
                        placement,
                        max_dim=2,
                        valid_split_axis=[y for y in other_shape if y != 1],
                    ):
                        _test_global_broadcast_tensors(
                            test_case, input_shape, other_shape, placement, x_sbp, y_sbp
                        )


if __name__ == "__main__":
    unittest.main()
