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

# TODO implement quadrati_interpolate op
def _quadratic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):

    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 < x2 else (x2, x1)
    if x1 == 0:
        t_new = -(g1 * (x2 ** 2)) / (2 * (f2 - f1 - g1 * x2))
    else:
        a = -(f1 - f2 - g1 * (x1 - x2)) / ((x1 - x2) ** 2)
        t_new = x1 - g1 / (2 * a)
    return min(xmax_bound, max(xmin_bound, t_new))


def _strong_wolfe(
    eval_closure, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    d_norm = d.abs().max()
    g = g.clone()
    f_new, g_new = eval_closure(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new > f_prev):
            search_area = [t_prev, t]
            search_area_f = [f_prev, f_new]
            search_area_g = [g_prev, g_new.clone()]
            search_area_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            search_area = [t]
            search_area_f = [f_new]
            search_area_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            search_area = [t_prev, t]
            search_area_f = [f_prev, f_new]
            search_area_g = [g_prev, g_new.clone()]
            search_area_gtd = [gtd_prev, gtd_new]

        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _quadratic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone()
        gtd_prev = gtd_new
        f_new, g_new = eval_closure(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1
    if ls_iter == max_ls:
        search_area = [0, t]
        search_area_f = [f, f_new]
        search_area_g = [g, g_new]

    # zoom
    low_pos, high_pos = (0, 1) if search_area_f[0] <= search_area_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:

        if abs(search_area[1] - search_area[0]) * d_norm < tolerance_change:
            break

        t = _quadratic_interpolate(
            search_area[0],
            search_area_f[0],
            search_area_gtd[0],
            search_area[1],
            search_area_f[1],
            search_area_gtd[1],
        )

        f_new, g_new = eval_closure(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= search_area_f[low_pos]:
            search_area[high_pos] = t
            search_area_f[high_pos] = f_new
            search_area_g[high_pos] = g_new.clone()
            search_area_gtd[high_pos] = gtd_new
            low_pos, high_pos = (
                (0, 1) if search_area_f[0] <= search_area_f[1] else (1, 0)
            )
        if abs(gtd_new) <= -c2 * gtd:
            done = True
        elif gtd_new * (search_area[high_pos] - search_area[low_pos]) >= 0:
            search_area[high_pos] = search_area[low_pos]
            search_area_f[high_pos] = search_area_f[low_pos]
            search_area_g[high_pos] = search_area_g[low_pos]
            search_area_gtd[high_pos] = search_area_gtd[low_pos]

        search_area[low_pos] = t
        search_area_f[low_pos] = f_new
        search_area_g[low_pos] = g_new.clone()
        search_area_gtd[low_pos] = gtd_new

    t = search_area[low_pos]
    f_new = search_area_f[low_pos]
    g_new = search_area_g[low_pos]
    return f_new, g_new, t, ls_func_evals


class LBFGS(Optimizer):
    """Implements LBFGS algorithm
    
    It has been propose in `On the limited memory BFGS method for large scale optimization`_.
    The implementation of the two-loop recursion proposed in `Updating Quasi-Newton Matrices with Limited Storage`_.
    
    The implementation of the strong_wolfe line search  proposed in `Numerical_Optimization_v2`
    
    This algorithm uses an estimated inverse Hessian matrix to steer its search through variable space and determine the optimal direction.
    
    The line search algorithm terminates with a step length that satisfies the strong Wolfe conditions.
    
    This optimizer only support one parameter group.        
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        max_iter (int,optional): max iteration per step (default: 20)
        max_eval (int,optional): max func evals per step (default: max_iter * 1.25)
        tolerance_grad (float, optional): termination tolerance on first order optimality (default 1e-7)
        tolerance_change (float, optional): termination tolerance on paramter changes (default: 1e-9)
        history_size (int,optional): paramter update history size (default: 100)
        line_search_fn (str,optional): line search function `strong_wolfe` or None (default: None)
        contiguous_params (bool, optional): whether to use contiguous ParamGroup 
            which puts all parameters of the same type, device and group into the
            same tensor and update them together. (default: False)
    .. _On the limited memory BFGS method for large scale optimization:
        https://dl.acm.org/doi/10.5555/3112655.3112866
            
    .. _Updating Quasi-Newton Matrices with Limited Storage:
        https://www.ams.org/journals/mcom/1980-35-151/S0025-5718-1980-0572855-7/S0025-5718-1980-0572855-7.pdf
    
    For example: 
    
    .. code-block:: python 
    
        # Assume net is a custom model. 
        lbfgs = flow.optim.LBFGS(net.parameters())
        
        for epoch in range (epochs):
            def closure():
                lbfgs.zero_grad()
                # Read data, Compute the loss and so on. 
                loss.backward()
                return loss
            lbfgs.step(closure)
                
    
    """

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
        contiguous_params: bool = False,
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
        options["contiguous_params"] = contiguous_params
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

    def _try_direction(self, closure, x, t, d):
        self._update(t, d)
        with flow.enable_grad():
            loss = float(closure())
        flag_grad = self._gather_flat_grad()
        for p, data in zip(self._params, x):
            p.copy_(data)
        return loss, flag_grad

    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
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
                        ls_func_evals = 1
                else:
                    assert (
                        line_search_fn == "strong_wolfe"
                    ), "only strong_wolfe is expected"
                    init_param = [p.clone() for p in self._params]

                    def eval_func(x, t, d):
                        return self._try_direction(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        eval_func, init_param, t, d, loss, flat_grad, gtd
                    )
                    self._update(t, d)

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
