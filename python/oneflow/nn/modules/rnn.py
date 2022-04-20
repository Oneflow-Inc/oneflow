"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import oneflow as flow
from oneflow import nn
from oneflow.nn import Module
from math import sqrt


class RNN(Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.RNN.html.

    Applies a multi-layer Elman RNN with \tanhtanh or \text{ReLU}ReLU non-linearity to an input sequence.

    For each element in the input sequence, each layer computes the following function:
    
    function:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
          state for each element in the batch. Defaults to zeros if not provided.

        where:
        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`
    
    .. note::
        For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view((seq_len, batch, num_directions, hidden_size))``.
    
    # For example:

    # .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> rnn = flow.nn.RNN(10, 20, 2)
        >>> input = flow.tensor(np.random.randn(5, 3, 10), dtype=flow.float32)
        >>> h0 = flow.tensor(np.random.randn(2, 3, 20), dtype=flow.float32)
        >>> output, hn = rnn(input, h0)
        >>> output.size()
        oneflow.Size([5, 3, 20])

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
    ):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        gate_size = hidden_size
        self.drop = nn.Dropout(self.dropout)

        if self.nonlinearity == "tanh":
            self.act = nn.Tanh()
        elif self.nonlinearity == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))

        for layer in range(num_layers):
            for direction in range(num_directions):

                real_hidden_size = hidden_size
                layer_input_size = (
                    input_size if layer == 0 else real_hidden_size * num_directions
                )

                # TODO: Modify after adding the stride attribute
                # w_ih = flow.nn.Parameter(flow.Tensor(gate_size, layer_input_size))
                # w_hh = flow.nn.Parameter(flow.Tensor(gate_size, real_hidden_size))
                # b_ih = flow.nn.Parameter(flow.Tensor(gate_size))
                # b_hh = flow.nn.Parameter(flow.Tensor(gate_size))

                w_ih = flow.nn.Parameter(flow.Tensor(layer_input_size, gate_size))
                w_hh = flow.nn.Parameter(flow.Tensor(real_hidden_size, gate_size))
                b_ih = flow.nn.Parameter(flow.Tensor(gate_size))
                b_hh = flow.nn.Parameter(flow.Tensor(gate_size))

                layer_params = ()

                if bias:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                else:
                    layer_params = (w_ih, w_hh)

                suffix = "_reverse" if direction == 1 else ""
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.uniform_(-stdv, stdv)

    def permute_tensor(self, input):
        return input.permute(1, 0, 2)

    def forward(self, input, h_0=None):
        if self.batch_first == False:
            input = self.permute_tensor(input)

        D = 2 if self.bidirectional else 1
        num_layers = self.num_layers
        batch_size, seq_len, _ = input.size()

        if h_0 is None:
            h_t = flow.zeros(
                (D * num_layers, batch_size, self.hidden_size),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            h_t = h_0

        if self.bidirectional:
            if h_0 is None:
                h_t_f = h_t[:num_layers, :, :]
                h_t_b = h_t[num_layers:, :, :]
            else:
                h_t_f = flow.cat(
                    [
                        h_t[l, :, :].unsqueeze(0)
                        for l in range(h_t.size(0))
                        if l % 2 == 0
                    ],
                    dim=0,
                )
                h_t_b = flow.cat(
                    [
                        h_t[l, :, :].unsqueeze(0)
                        for l in range(h_t.size(0))
                        if l % 2 != 0
                    ],
                    dim=0,
                )
        else:
            h_t_f = h_t

        layer_hidden = []

        for layer in range(self.num_layers):
            hidden_seq_f = []
            if self.bidirectional:
                hidden_seq_b = []

            hid_t_f = h_t_f[layer, :, :]
            if self.bidirectional:
                hid_t_b = h_t_b[layer, :, :]

            for t in range(seq_len):
                if layer == 0:
                    x_t_f = input[:, t, :]
                    if self.bidirectional:
                        x_t_b = input[:, seq_len - 1 - t, :]
                else:
                    x_t_f = hidden_seq[:, t, :]
                    if self.bidirectional:
                        x_t_b = hidden_seq[:, seq_len - 1 - t, :]

                # TODO: Modify after adding the stride attribute
                # hy1_f = flow.matmul(
                #     x_t_f,
                #     getattr(self, "weight_ih_l{}{}".format(layer, "")).permute(1, 0),
                # )
                # hy2_f = flow.matmul(
                #     hid_t_f,
                #     getattr(self, "weight_hh_l{}{}".format(layer, "")).permute(1, 0),
                # )

                hy1_f = flow.matmul(
                    x_t_f, getattr(self, "weight_ih_l{}{}".format(layer, "")),
                )
                hy2_f = flow.matmul(
                    hid_t_f, getattr(self, "weight_hh_l{}{}".format(layer, "")),
                )

                if self.bias:
                    hy1_f += getattr(self, "bias_ih_l{}{}".format(layer, ""))
                    hy2_f += getattr(self, "bias_hh_l{}{}".format(layer, ""))
                hid_t_f = self.act(hy1_f + hy2_f)

                hidden_seq_f.append(hid_t_f.unsqueeze(1))

                if self.bidirectional:

                    # TODO：Modify after adding the stride attribute
                    # hy1_b = flow.matmul(
                    #     x_t_b,
                    #     getattr(
                    #         self, "weight_ih_l{}{}".format(layer, "_reverse")
                    #     ).permute(1, 0),
                    # )
                    # hy2_b = flow.matmul(
                    #     hid_t_b,
                    #     getattr(
                    #         self, "weight_hh_l{}{}".format(layer, "_reverse")
                    #     ).permute(1, 0),
                    # )

                    hy1_b = flow.matmul(
                        x_t_b,
                        getattr(self, "weight_ih_l{}{}".format(layer, "_reverse")),
                    )
                    hy2_b = flow.matmul(
                        hid_t_b,
                        getattr(self, "weight_hh_l{}{}".format(layer, "_reverse")),
                    )
                    if self.bias:
                        hy1_b += getattr(
                            self, "bias_ih_l{}{}".format(layer, "_reverse")
                        )
                        hy2_b += getattr(
                            self, "bias_hh_l{}{}".format(layer, "_reverse")
                        )
                    hid_t_b = self.act(hy1_b + hy2_b)

                    hidden_seq_b.insert(0, hid_t_b.unsqueeze(1))

            hidden_seq_f = flow.cat(hidden_seq_f, dim=1)
            if self.bidirectional:
                hidden_seq_b = flow.cat(hidden_seq_b, dim=1)

            if self.dropout != 0 and layer != self.num_layers - 1:
                hidden_seq_f = self.drop(hidden_seq_f)
                if self.bidirectional:
                    hidden_seq_b = self.drop(hidden_seq_b)

            if self.bidirectional:
                hidden_seq = flow.cat([hidden_seq_f, hidden_seq_b], dim=2)
            else:
                hidden_seq = hidden_seq_f

            if self.bidirectional:
                h_t = flow.cat([hid_t_f.unsqueeze(0), hid_t_b.unsqueeze(0)], dim=0)
            else:
                h_t = hid_t_f.unsqueeze(0)

            layer_hidden.append(h_t)

        h_t = flow.cat(layer_hidden, dim=0)

        if self.batch_first == False:
            hidden_seq = self.permute_tensor(hidden_seq)

        return hidden_seq, h_t


class GRU(Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/_modules/torch/nn/modules/rnn.html#GRU.

    Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

    For each element in the input sequence, each layer computes the following
    
    function:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}
    
    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence. 
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
          state for each element in the batch. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the GRU, for each `t`. If a
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    # For example:

    # .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> rnn = flow.nn.GRU(10, 20, 2)
        >>> input = flow.tensor(np.random.randn(5, 3, 10), dtype=flow.float32)
        >>> h0 = flow.tensor(np.random.randn(2, 3, 20), dtype=flow.float32)
        >>> output, hn = rnn(input, h0)
        >>> output.size()
        oneflow.Size([5, 3, 20])
    
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        gate_size = 3 * hidden_size
        self.drop = nn.Dropout(self.dropout)

        for layer in range(num_layers):
            for direction in range(num_directions):

                real_hidden_size = hidden_size
                layer_input_size = (
                    input_size if layer == 0 else real_hidden_size * num_directions
                )

                # TODO: Modify after adding the stride attribute
                # w_ih = flow.nn.Parameter(flow.Tensor(gate_size, layer_input_size))
                # w_hh = flow.nn.Parameter(flow.Tensor(gate_size, real_hidden_size))
                # b_ih = flow.nn.Parameter(flow.Tensor(gate_size))
                # b_hh = flow.nn.Parameter(flow.Tensor(gate_size))

                w_ih = flow.nn.Parameter(flow.Tensor(layer_input_size, gate_size))
                w_hh = flow.nn.Parameter(flow.Tensor(real_hidden_size, gate_size))
                b_ih = flow.nn.Parameter(flow.Tensor(gate_size))
                b_hh = flow.nn.Parameter(flow.Tensor(gate_size))

                layer_params = ()

                if bias:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                else:
                    layer_params = (w_ih, w_hh)

                suffix = "_reverse" if direction == 1 else ""
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.uniform_(-stdv, stdv)

    def permute_tensor(self, input):
        return input.permute(1, 0, 2)

    def forward(self, input, h_0=None):
        if self.batch_first == False:
            input = self.permute_tensor(input)
        D = 2 if self.bidirectional else 1
        num_layers = self.num_layers
        batch_size, seq_len, _ = input.size()

        if h_0 is None:
            h_t = flow.zeros(
                (D * num_layers, batch_size, self.hidden_size),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            h_t = h_0

        if self.bidirectional:
            if h_0 is None:
                h_t_f = h_t[:num_layers, :, :]
                h_t_b = h_t[num_layers:, :, :]
            else:
                h_t_f = flow.cat(
                    [
                        h_t[l, :, :].unsqueeze(0)
                        for l in range(h_t.size(0))
                        if l % 2 == 0
                    ],
                    dim=0,
                )
                h_t_b = flow.cat(
                    [
                        h_t[l, :, :].unsqueeze(0)
                        for l in range(h_t.size(0))
                        if l % 2 != 0
                    ],
                    dim=0,
                )
        else:
            h_t_f = h_t

        layer_hidden = []

        for layer in range(self.num_layers):
            hidden_seq_f = []
            if self.bidirectional:
                hidden_seq_b = []

            hid_t_f = h_t_f[layer, :, :]
            if self.bidirectional:
                hid_t_b = h_t_b[layer, :, :]

            for t in range(seq_len):
                if layer == 0:
                    x_t_f = input[:, t, :]
                    if self.bidirectional:
                        x_t_b = input[:, seq_len - 1 - t, :]
                else:
                    x_t_f = hidden_seq[:, t, :]
                    if self.bidirectional:
                        x_t_b = hidden_seq[:, seq_len - 1 - t, :]

                # TODO: Modify after adding the stride attribute
                # gi_f = flow.matmul(
                #     x_t_f,
                #     getattr(self, "weight_ih_l{}{}".format(layer, "")).permute(1, 0),
                # )
                # gh_f = flow.matmul(
                #     hid_t_f,
                #     getattr(self, "weight_hh_l{}{}".format(layer, "")).permute(1, 0),
                # )

                gi_f = flow.matmul(
                    x_t_f, getattr(self, "weight_ih_l{}{}".format(layer, "")),
                )
                gh_f = flow.matmul(
                    hid_t_f, getattr(self, "weight_hh_l{}{}".format(layer, "")),
                )
                if self.bias:
                    gi_f += getattr(self, "bias_ih_l{}{}".format(layer, ""))
                    gh_f += getattr(self, "bias_hh_l{}{}".format(layer, ""))

                i_r_f, i_i_f, i_n_f = gi_f.chunk(3, dim=1)
                h_r_f, h_i_f, h_n_f = gh_f.chunk(3, dim=1)

                resetgate_f = flow.sigmoid(i_r_f + h_r_f)
                inputgate_f = flow.sigmoid(i_i_f + h_i_f)
                newgate_f = flow.tanh(i_n_f + resetgate_f * h_n_f)

                hid_t_f = newgate_f + inputgate_f * (hid_t_f - newgate_f)

                hidden_seq_f.append(hid_t_f.unsqueeze(1))

                if self.bidirectional:

                    # TODO：Modify after adding the stride attribute
                    # gi_b = flow.matmul(
                    #     x_t_b,
                    #     getattr(
                    #         self, "weight_ih_l{}{}".format(layer, "_reverse")
                    #     ).permute(1, 0),
                    # )
                    # gh_b = flow.matmul(
                    #     hid_t_b,
                    #     getattr(
                    #         self, "weight_hh_l{}{}".format(layer, "_reverse")
                    #     ).permute(1, 0),
                    # )

                    gi_b = flow.matmul(
                        x_t_b,
                        getattr(self, "weight_ih_l{}{}".format(layer, "_reverse")),
                    )
                    gh_b = flow.matmul(
                        hid_t_b,
                        getattr(self, "weight_hh_l{}{}".format(layer, "_reverse")),
                    )
                    if self.bias:
                        gi_b += getattr(self, "bias_ih_l{}{}".format(layer, "_reverse"))
                        gh_b += getattr(self, "bias_hh_l{}{}".format(layer, "_reverse"))

                    i_r_b, i_i_b, i_n_b = gi_b.chunk(3, dim=1)
                    h_r_b, h_i_b, h_n_b = gh_b.chunk(3, dim=1)

                    resetgate_b = flow.sigmoid(i_r_b + h_r_b)
                    inputgate_b = flow.sigmoid(i_i_b + h_i_b)
                    newgate_b = flow.tanh(i_n_b + resetgate_b * h_n_b)

                    hid_t_b = newgate_b + inputgate_b * (hid_t_b - newgate_b)

                    hidden_seq_b.insert(0, hid_t_b.unsqueeze(1))

            hidden_seq_f = flow.cat(hidden_seq_f, dim=1)
            if self.bidirectional:
                hidden_seq_b = flow.cat(hidden_seq_b, dim=1)

            if self.dropout != 0 and layer != self.num_layers - 1:
                hidden_seq_f = self.drop(hidden_seq_f)
                if self.bidirectional:
                    hidden_seq_b = self.drop(hidden_seq_b)

            if self.bidirectional:
                hidden_seq = flow.cat([hidden_seq_f, hidden_seq_b], dim=2)
            else:
                hidden_seq = hidden_seq_f

            if self.bidirectional:
                h_t = flow.cat([hid_t_f.unsqueeze(0), hid_t_b.unsqueeze(0)], dim=0)
            else:
                h_t = hid_t_f.unsqueeze(0)

            layer_hidden.append(h_t)

        h_t = flow.cat(layer_hidden, dim=0)

        if self.batch_first == False:
            hidden_seq = self.permute_tensor(hidden_seq)

        return hidden_seq, h_t


class LSTM(nn.Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/_modules/torch/nn/modules/rnn.html#LSTM.

    Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

    For each element in the input sequence, each layer computes the following
    
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state of each layer will be multiplied by a learnable projection
    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output
    of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact
    dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

    Inputs: input, (h_0, c_0)
        * **input**: tensor of shape :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          initial hidden state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **c_0**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          initial cell state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{cell} ={} & \text{hidden\_size} \\
                H_{out} ={} & \text{proj\_size if } \text{proj\_size}>0 \text{ otherwise hidden\_size} \\
            \end{aligned}

    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          final hidden state for each element in the batch.
        * **c_n**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          final cell state for each element in the batch.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``
            was specified, the shape will be `(4*hidden_size, proj_size)`.
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
        weight_hr_l[k] : the learnable projection weights of the :math:`\text{k}^{th}` layer
            of shape `(proj_size, hidden_size)`. Only present when ``proj_size > 0`` was
            specified.

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    # For example:

    # .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> rnn = flow.nn.LSTM(10, 20, 2)
        >>> input = flow.tensor(np.random.randn(5, 3, 10), dtype=flow.float32)
        >>> h0 = flow.tensor(np.random.randn(2, 3, 20), dtype=flow.float32)
        >>> c0 = flow.tensor(np.random.randn(2, 3, 20), dtype=flow.float32)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
        >>> output.size()
        oneflow.Size([5, 3, 20])
        
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        gate_size = 4 * hidden_size
        self.proj_size = proj_size
        self.drop = nn.Dropout(self.dropout)

        if proj_size < 0:
            raise ValueError(
                "proj_size should be a positive integer or zero to disable projections"
            )
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        for layer in range(num_layers):
            for direction in range(num_directions):

                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                layer_input_size = (
                    input_size if layer == 0 else real_hidden_size * num_directions
                )

                # TODO：Modify after adding the stride attribute
                # w_ih = flow.nn.Parameter(flow.Tensor(gate_size, layer_input_size))
                # w_hh = flow.nn.Parameter(flow.Tensor(gate_size, real_hidden_size))
                # b_ih = flow.nn.Parameter(flow.Tensor(gate_size))
                # b_hh = flow.nn.Parameter(flow.Tensor(gate_size))

                w_ih = flow.nn.Parameter(flow.Tensor(layer_input_size, gate_size))
                w_hh = flow.nn.Parameter(flow.Tensor(real_hidden_size, gate_size))
                b_ih = flow.nn.Parameter(flow.Tensor(gate_size))
                b_hh = flow.nn.Parameter(flow.Tensor(gate_size))

                layer_params = ()

                if self.proj_size == 0:
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)
                else:

                    # TODO: Modify after adding the stride attribute
                    # w_hr = flow.nn.Parameter(flow.Tensor(proj_size, hidden_size))

                    w_hr = flow.nn.Parameter(flow.Tensor(hidden_size, proj_size))

                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
                    else:
                        layer_params = (w_ih, w_hh, w_hr)

                suffix = "_reverse" if direction == 1 else ""
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                if self.proj_size > 0:
                    param_names += ["weight_hr_l{}{}"]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.uniform_(-stdv, stdv)

    def permute_tensor(self, input):
        return input.permute(1, 0, 2)

    def forward(self, input, h_0=None):
        if self.batch_first == False:
            input = self.permute_tensor(input)
        D = 2 if self.bidirectional else 1
        num_layers = self.num_layers
        batch_size, seq_len, _ = input.size()

        if h_0 is None:
            real_hidden_size = (
                self.proj_size if self.proj_size > 0 else self.hidden_size
            )
            h_t = flow.zeros(
                (D * num_layers, batch_size, real_hidden_size),
                dtype=input.dtype,
                device=input.device,
            )
            c_t = flow.zeros(
                (D * num_layers, batch_size, self.hidden_size),
                dtype=input.dtype,
                device=input.device,
            )
            h_0 = (h_t, c_t)
        else:
            h_t, c_t = h_0

        if self.bidirectional:
            if h_0 is None:
                h_t_f = h_t[:num_layers, :, :]
                h_t_b = h_t[num_layers:, :, :]
                c_t_f = c_t[:num_layers, :, :]
                c_t_b = c_t[num_layers:, :, :]
            else:
                h_t_f = flow.cat(
                    [
                        h_t[l, :, :].unsqueeze(0)
                        for l in range(h_t.size(0))
                        if l % 2 == 0
                    ],
                    dim=0,
                )
                h_t_b = flow.cat(
                    [
                        h_t[l, :, :].unsqueeze(0)
                        for l in range(h_t.size(0))
                        if l % 2 != 0
                    ],
                    dim=0,
                )
                c_t_f = flow.cat(
                    [
                        c_t[l, :, :].unsqueeze(0)
                        for l in range(c_t.size(0))
                        if l % 2 == 0
                    ],
                    dim=0,
                )
                c_t_b = flow.cat(
                    [
                        c_t[l, :, :].unsqueeze(0)
                        for l in range(c_t.size(0))
                        if l % 2 != 0
                    ],
                    dim=0,
                )
        else:
            h_t_f = h_t
            c_t_f = c_t

        layer_hidden = []
        layer_cell = []

        for layer in range(self.num_layers):

            hidden_seq_f = []
            if self.bidirectional:
                hidden_seq_b = []

            hid_t_f = h_t_f[layer, :, :]
            h_c_t_f = c_t_f[layer, :, :]
            if self.bidirectional:
                hid_t_b = h_t_b[layer, :, :]
                h_c_t_b = c_t_b[layer, :, :]

            for t in range(seq_len):
                if layer == 0:
                    x_t_f = input[:, t, :]
                    if self.bidirectional:
                        x_t_b = input[:, seq_len - 1 - t, :]
                else:
                    x_t_f = hidden_seq[:, t, :]
                    if self.bidirectional:
                        x_t_b = hidden_seq[:, seq_len - 1 - t, :]

                # TODO: Modify after adding the stride attribute
                # gi_f = flow.matmul(
                #     x_t_f,
                #     getattr(self, "weight_ih_l{}{}".format(layer, "")).permute(1, 0),
                # )
                # gh_f = flow.matmul(
                #     hid_t_f,
                #     getattr(self, "weight_hh_l{}{}".format(layer, "")).permute(1, 0),
                # )

                gi_f = flow.matmul(
                    x_t_f, getattr(self, "weight_ih_l{}{}".format(layer, "")),
                )
                gh_f = flow.matmul(
                    hid_t_f, getattr(self, "weight_hh_l{}{}".format(layer, "")),
                )
                if self.bias:
                    gi_f += getattr(self, "bias_ih_l{}{}".format(layer, ""))
                    gh_f += getattr(self, "bias_hh_l{}{}".format(layer, ""))
                gates_f = gi_f + gh_f
                ingate_f, forgetgate_f, cellgate_f, outgate_f = gates_f.chunk(4, dim=1)
                ingate_f = flow.sigmoid(ingate_f)
                forgetgate_f = flow.sigmoid(forgetgate_f)
                cellgate_f = flow.tanh(cellgate_f)
                outgate_f = flow.sigmoid(outgate_f)
                h_c_t_f = (forgetgate_f * h_c_t_f) + (ingate_f * cellgate_f)
                hid_t_f = outgate_f * flow.tanh(h_c_t_f)
                if self.proj_size > 0:

                    # TODO：Modify after adding the stride attribute
                    # hid_t_f = flow.matmul(
                    #     hid_t_f,
                    #     getattr(self, "weight_hr_l{}{}".format(layer, "")).permute(
                    #         1, 0
                    #     ),
                    # )

                    hid_t_f = flow.matmul(
                        hid_t_f, getattr(self, "weight_hr_l{}{}".format(layer, ""))
                    )
                hidden_seq_f.append(hid_t_f.unsqueeze(1))

                if self.bidirectional:

                    # TODO：Modify after adding the stride attribute
                    # gi_b = flow.matmul(
                    #     x_t_b,
                    #     getattr(
                    #         self, "weight_ih_l{}{}".format(layer, "_reverse")
                    #     ).permute(1, 0),
                    # )
                    # gh_b = flow.matmul(
                    #     hid_t_b,
                    #     getattr(
                    #         self, "weight_hh_l{}{}".format(layer, "_reverse")
                    #     ).permute(1, 0),
                    # )

                    gi_b = flow.matmul(
                        x_t_b,
                        getattr(self, "weight_ih_l{}{}".format(layer, "_reverse")),
                    )
                    gh_b = flow.matmul(
                        hid_t_b,
                        getattr(self, "weight_hh_l{}{}".format(layer, "_reverse")),
                    )

                    if self.bias:
                        gi_b += getattr(self, "bias_ih_l{}{}".format(layer, "_reverse"))
                        gh_b += getattr(self, "bias_hh_l{}{}".format(layer, "_reverse"))
                    gates_b = gi_b + gh_b
                    ingate_b, forgetgate_b, cellgate_b, outgate_b = gates_b.chunk(
                        4, dim=1
                    )
                    ingate_b = flow.sigmoid(ingate_b)
                    forgetgate_b = flow.sigmoid(forgetgate_b)
                    cellgate_b = flow.tanh(cellgate_b)
                    outgate_b = flow.sigmoid(outgate_b)
                    h_c_t_b = (forgetgate_b * h_c_t_b) + (ingate_b * cellgate_b)
                    hid_t_b = outgate_b * flow.tanh(h_c_t_b)
                    if self.proj_size > 0:

                        # TODO：Modify after adding the stride attribute
                        # hid_t_b = flow.matmul(
                        #     hid_t_b,
                        #     getattr(
                        #         self, "weight_hr_l{}{}".format(layer, "_reverse")
                        #     ).permute(1, 0),
                        # )

                        hid_t_b = flow.matmul(
                            hid_t_b,
                            getattr(self, "weight_hr_l{}{}".format(layer, "_reverse")),
                        )
                    hidden_seq_b.insert(0, hid_t_b.unsqueeze(1))

            hidden_seq_f = flow.cat(hidden_seq_f, dim=1)
            if self.bidirectional:
                hidden_seq_b = flow.cat(hidden_seq_b, dim=1)

            if self.dropout != 0 and layer != self.num_layers - 1:
                hidden_seq_f = self.drop(hidden_seq_f)
                if self.bidirectional:
                    hidden_seq_b = self.drop(hidden_seq_b)

            if self.bidirectional:
                hidden_seq = flow.cat([hidden_seq_f, hidden_seq_b], dim=2)
            else:
                hidden_seq = hidden_seq_f

            if self.bidirectional:
                h_t = flow.cat([hid_t_f.unsqueeze(0), hid_t_b.unsqueeze(0)], dim=0)
                c_t = flow.cat([h_c_t_f.unsqueeze(0), h_c_t_b.unsqueeze(0)], dim=0)
            else:
                h_t = hid_t_f.unsqueeze(0)
                c_t = h_c_t_f.unsqueeze(0)

            layer_hidden.append(h_t)
            layer_cell.append(c_t)

        h_t = flow.cat(layer_hidden, dim=0)
        c_t = flow.cat(layer_cell, dim=0)

        if self.batch_first == False:
            hidden_seq = self.permute_tensor(hidden_seq)

        return hidden_seq, (h_t, c_t)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
