"""Test contiguous parameter functions."""
from copy import deepcopy
import os

import oneflow
import numpy as np
from oneflow import nn

from oneflow.nn.optimizer.contiguous_params import ContiguousParams


def test_equal_optimizer_update(device):
    if device == "cuda" and not oneflow.cuda.is_available():
        print("No GPU available, skipping GPU test.")
        return
    """Verify that the parameters are the same after a few updates."""
    x = oneflow.randn(1, 8).to(device)

    model_ref = nn.Sequential(*[nn.Linear(8, 8) for i in range(10)])
    model_ref = model_ref.to(device)
    optimizer = oneflow.optim.SGD(model_ref.parameters(), lr=1e-3)
    
    model_c = deepcopy(model_ref)
    parameters_c = ContiguousParams(model_c.parameters())
    optimizer_c = oneflow.optim.SGD(parameters_c.contiguous(), lr=1e-3)

    for model, optimizer in zip([model_ref, model_c], [optimizer, optimizer_c]):
        for step in range(5):
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Verify that the model/optimizer did not modify the data or grad handle.

    # Verify that both models applied the same parameter updates.
    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        assert np.allclose(p1.numpy(), p2.numpy(), atol=1e-06)

test_equal_optimizer_update('cuda')

"""
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_buffer_invalidation_detection(device):
    if device == "cuda" and not torch.cuda.is_available():
        print("No GPU available, skipping GPU test.")
        return
    model = nn.Linear(8, 8)
    parameters = ContiguousParams(model.parameters())
    assert parameters.buffer_is_valid()
    # Invalidate the buffer.
    model.weight.data = model.weight + 4
    assert not parameters.buffer_is_valid()
    with pytest.raises(ValueError):
      parameters.assert_buffer_is_valid()


def test_distributed_data_parallel():
    np.random.seed(0)
    # Create 20 samples with 10 features, label one out of 5 classes.
    data_X = torch.as_tensor(np.random.randn(20, 10), dtype=torch.float32)
    data_y = torch.as_tensor(np.random.choice(5, (20)), dtype=torch.int64)
    
    class Model(pytorch_lightning.LightningModule):

        def __init__(self, use_contiguous):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(10, 10),
                                nn.ReLU(),
                                nn.Linear(10, 5))
            self.use_contiguous = use_contiguous
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.dataset = TensorDataset(data_X, data_y)
            self.contiguous_params = None
            self.optimizer = None
        
        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, target = batch
            prediction = self(x)
            loss_value = self.loss_fn(prediction, target)
            return {'loss': loss_value}

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.dataset,
                                               batch_size=2,
                                               shuffle=False)

        def configure_optimizers(self):
            if self.use_contiguous:
                self.contiguous_params = ContiguousParams(self.parameters())
                params = self.contiguous_params.contiguous()
            else:
                params = self.model.parameters()
            self.optimizer = torch.optim.SGD(params, lr=1e-3)
            return self.optimizer

    
    model_ref = Model(use_contiguous=False)
    initial_configuration = deepcopy(model_ref.state_dict())
    
    model_c = Model(use_contiguous=True)
    model_c.load_state_dict(initial_configuration)

    port = 1234
    for i, model in enumerate([model_ref, model_c]):
        # Choose different ports to prevent
        # RuntimeError("Address already in use.").
        os.environ['MASTER_PORT'] = str(port + i)
        trainer = pytorch_lightning.Trainer(distributed_backend="ddp", max_epochs=1, gpus=[0])
        trainer.fit(model)
        # Make sure the optimizer did update the weights.
        for p1, p2 in zip(model.parameters(), initial_configuration.values()):
            assert not torch.allclose(p1.data, p2.data, atol=1e-06)
    
    
    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        assert torch.allclose(p1.data, p2.data, atol=1e-06)
"""
