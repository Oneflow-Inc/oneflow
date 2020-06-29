import os
import numpy
from absl import app
from absl.testing import absltest
import oneflow as flow

flow.unittest.register_test_cases(
    scope=globals(),
    directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), "nodes"),
    filter_by_num_nodes=lambda x: True,
    base_class=absltest.TestCase,
)


def main(argv):
    flow.env.init()
    absltest.main()


if __name__ == "__main__":
    app.run(main)
