
class LossScalePolicy:
    def generate_conf_for_graph(self, train_conf):
        raise NotImplementedError()


class StaticLossScalePolicy(LossScalePolicy):
    def __init__(self, loss_scale_factor: float):
        super().__init__()
        self.loss_scale_factor = loss_scale_factor

    def generate_conf_for_graph(self, train_conf):
        train_conf.loss_scale_factor = self.loss_scale_factor


class DynamicLossScalePolicy(LossScalePolicy):
    def __init__(
        self, initial_loss_scale=2 ** 30, increment_period=2000, multiplier=2.0
    ):
        super().__init__()
        self.initial_loss_scale = initial_loss_scale
        self.increment_period = increment_period
        self.multiplier = multiplier

    def generate_conf_for_graph(self, train_conf):
        train_conf.mutable_dynamic_loss_scale_policy().set_initial_loss_scale(
            self.initial_loss_scale
        )
        train_conf.mutable_dynamic_loss_scale_policy().set_increment_period(
            self.increment_period
        )
        train_conf.mutable_dynamic_loss_scale_policy().set_multiplier(self.multiplier)