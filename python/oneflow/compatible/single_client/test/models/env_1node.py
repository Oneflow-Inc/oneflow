from oneflow.compatible import single_client as flow


def Init():
    flow.env.init()
