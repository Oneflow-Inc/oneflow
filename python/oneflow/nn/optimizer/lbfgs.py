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
from typing import Callable, Dict, Iterator, List, Tuple, Union
from functools import reduce
from oneflow.optim.optimizer import Optimizer
from oneflow.nn.parameter import Parameter
import oneflow as flow


class LBFGS(Optimizer):
    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        max_iter: int = 20,
        max_eval: int = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn=None,
    ):
        if max_eval is None:
            max_eval = max_iter * 1.25
        options = dict()
        options["lr"] = lr
        options["max_iter"] = max_iter
        options["max_eval"] = max_eval
        options["tolerance_grad"] = tolerance_grad
        options["tolerance_change"] = tolerance_change
        options["history_size"] = history_size
        options["line_search_fn"] = line_search_fn
        super().__init__(params, options)
        assert (
            len(self.param_groups) == 1
        ), "LBFGS not support parameter groups (there can be only one)"
        param_group = self.param_groups[0]
        if param_group["contiguous_params"]:
            param_list = param_group.contiguous_parameters
        else:
            param_list = param_group.parameters
        for param in param_list:
            assert param.is_leaf, "parameters must be leaf tensor"
        self._params = param_list
        self._numel_cache = None

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return flow.cat(views, 0)

    def _numel(self):
        # get parameters total numel
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda totnumel, p: totnumel + p.numel(), self._params, 0,
            )
        return self._numel_cache

    def _update(self, step_size, direction):
        # update parameters
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(direction[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def step(self, closure: Callable = None):
        with flow.no_grad():

            assert closure != None, "closure must not be None"
            param_group = self.param_groups[0]
            lr = param_group["lr"]
            max_iter = param_group["max_iter"]
            max_eval = param_group["max_eval"]
            tolerance_grad = param_group["tolerance_grad"]
            tolerance_change = param_group["tolerance_change"]
            line_search_fn = param_group["line_search_fn"]
            history_size = param_group["history_size"]

            state = self.state[self._params[0]]
            state.setdefault("func_evals", 0)
            state.setdefault("n_iter", 0)
            with flow.enable_grad():
                origin_loss = closure()
            loss = float(origin_loss)
            current_evals = 1
            state["func_evals"] += 1

            flat_grad = self._gather_flat_grad()

            if flat_grad.abs().max() <= tolerance_grad:
                return origin_loss

            # prev state
            d = state.get("d")
            t = state.get("t")
            old_diffs = state.get("old_diffs")
            old_step_size = state.get("old_step_size")
            ro = state.get("ro")
            H_diag = state.get("H_diag")
            prev_flat_grad = state.get("prev_flat_grad")
            prev_loss = state.get("prev_loss")

            n_iter = 0

            while n_iter < max_iter:
                n_iter += 1
                state["n_iter"] += 1

                # compute direction
                if state["n_iter"] == 1:
                    d = flat_grad.neg()
                    old_diffs = []
                    old_step_size = []
                    ro = []
                    H_diag = 1
                else:
                    y = flat_grad.sub(prev_flat_grad)
                    s = d.mul(t)
                    ys = y.dot(s)
                    # ys must be positive
                    if ys > 1e-10:
                        if len(old_diffs) == history_size:
                            old_diffs.pop(0)
                            old_step_size.pop(0)
                            ro.pop(0)
                        old_diffs.append(y)
                        old_step_size.append(s)
                        ro.append(1.0 / ys)
                        H_diag = ys / y.dot(y)

                    num_old = len(old_diffs)

                    if "alpha" not in state:
                        state["alpha"] = [None] * history_size
                    alpha = state["alpha"]

                    q = flat_grad.neg()
                    # import pdb; pdb.set_trace()
                    for i in range(num_old - 1, -1, -1):
                        alpha[i] = old_step_size[i].dot(q) * ro[i]
                        q.add_(old_diffs[i], alpha=-alpha[i])

                    d = q.mul(H_diag)
                    for i in range(num_old):
                        beta_i = old_diffs[i].dot(d) * ro[i]
                        d.add_(old_step_size[i], alpha=alpha[i] - beta_i)

                # compute step size
                if prev_flat_grad is None:
                    prev_flat_grad = flat_grad.clone()
                else:
                    prev_flat_grad.copy_(flat_grad)

                prev_loss = loss

                if state["n_iter"] == 1:
                    t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
                else:
                    t = lr

                gtd = flat_grad.dot(d)
                if gtd > -tolerance_change:
                    break

                ls_func_evals = 0
                if line_search_fn is None:
                    self._update(t, d)
                    if n_iter != max_iter:
                        with flow.enable_grad():
                            loss = float(closure())
                        flat_grad = self._gather_flat_grad()
                        ls_func_evals += 1

                current_evals += ls_func_evals
                state["func_evals"] += ls_func_evals

                if n_iter == max_iter:
                    break

                if current_evals >= max_eval:
                    break

                if flat_grad.abs().max() <= tolerance_grad:
                    break

                if d.mul(t).abs().max() <= tolerance_change:
                    break

                if abs(loss - prev_loss) < tolerance_change:
                    break

            state["d"] = d
            state["t"] = t
            state["old_diffs"] = old_diffs
            state["old_step_size"] = old_step_size
            state["ro"] = ro
            state["prev_flat_grad"] = prev_flat_grad
            state["prev_loss"] = prev_loss
            state["H_diag"] = H_diag
            return origin_loss
