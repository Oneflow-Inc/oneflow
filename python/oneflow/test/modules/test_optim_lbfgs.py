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
import tempfile
import unittest
from collections import OrderedDict
import random as random_util

import numpy as np
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import random_device, random_bool
import oneflow as flow
from oneflow.nn.parameter import Parameter
from collections import defaultdict


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

    d_norm = max(map(abs, d))
    g = np.copy(g)
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
            search_area_g = [g_prev, np.copy(g_new)]
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
            search_area_g = [g_prev, np.copy(g_new)]
            search_area_gtd = [gtd_prev, gtd_new]

        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _quadratic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )
        t_prev = tmp
        f_prev = f_new
        g_prev = np.copy(g_new)
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
            search_area_g[high_pos] = np.copy(g_new)
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
        search_area_g[low_pos] = np.copy(g_new)
        search_area_gtd[low_pos] = gtd_new

    t = search_area[low_pos]
    f_new = search_area_f[low_pos]
    g_new = search_area_g[low_pos]
    return f_new, g_new, t, ls_func_evals


def compare_with_numpy_lbfgs(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    max_iter,
    max_eval,
    tolerance_grad,
    tolerance_change,
    history_size,
    line_search_fn,
    reload_state_step,
    save_load_by_pickle,
    contiguous_params,
    tensor_num,
    use_float64,
):
    random_grad_seq = []
    init_value_seq = []
    if use_float64:
        npType = np.float64
        flowType = flow.float64
        flow.set_default_tensor_type(flow.DoubleTensor)
    else:
        npType = np.float32
        flowType = flow.float32
        flow.set_default_tensor_type(flow.FloatTensor)
    for _ in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(npType))
    for _ in range(tensor_num):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(npType))

    def train_by_oneflow():
        x = []
        for i in range(tensor_num):
            x.append(
                Parameter(
                    flow.tensor(
                        init_value_seq[i], device=flow.device(device), dtype=flowType
                    )
                )
            )

        lbfgs = flow.optim.LBFGS(
            [{"params": x}],
            lr=learning_rate,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
            contiguous_params=contiguous_params,
        )

        def compute_loss(grad):
            loss = 0.0
            for i in range(tensor_num):
                grad_tensor = flow.tensor(
                    grad[i],
                    dtype=flowType,
                    requires_grad=False,
                    device=flow.device(device),
                )
                loss += flow.sum(x[i] * x[i] * grad_tensor)
            loss.backward()
            return loss

        def train_one_iter(grad):
            def closure():
                lbfgs.zero_grad()
                loss = compute_loss(grad)
                return loss

            return lbfgs.step(closure)

        for i in range(train_iters):
            train_one_iter(random_grad_seq)
            if i == reload_state_step:
                state_dict = lbfgs.state_dict()
                lbfgs = flow.optim.LBFGS(
                    [{"params": x,}], contiguous_params=contiguous_params
                )
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                lbfgs.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        def compute_loss(param, grad):
            loss = 0.0
            loss += np.sum(param * param * grad)
            return loss

        x = np.concatenate(init_value_seq)

        def np_train_one_iter(x, state, init_grad):
            flat_grad = 2 * x * init_grad
            if max(map(abs, flat_grad)) <= tolerance_grad:
                return x
            loss = compute_loss(x, init_grad)
            current_evals = 1
            state["func_evals"] += 1
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
                if state["n_iter"] == 1:
                    d = -flat_grad
                    old_diffs = []
                    old_step_size = []
                    ro = []
                    H_diag = 1
                else:
                    y = flat_grad - prev_flat_grad
                    s = d * t
                    ys = y.dot(s)
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

                    q = -flat_grad
                    for i in range(num_old - 1, -1, -1):
                        alpha[i] = old_step_size[i].dot(q) * ro[i]
                        q += old_diffs[i] * -alpha[i]
                    d = q * H_diag
                    for i in range(num_old):
                        beta_i = old_diffs[i].dot(d) * ro[i]
                        d += old_step_size[i] * (alpha[i] - beta_i)

                prev_flat_grad = np.copy(flat_grad)
                prev_loss = loss
                if state["n_iter"] == 1:
                    t = min(1.0, 1.0 / np.sum(np.abs(flat_grad))) * learning_rate
                else:
                    t = learning_rate
                gtd = flat_grad.dot(d)
                if gtd > -tolerance_change:
                    break

                ls_func_evals = 0
                if line_search_fn is None:
                    x += t * d
                    if n_iter != max_iter:
                        loss = float(compute_loss(x, init_grad))
                        ls_func_evals = 1
                        flat_grad = 2 * x * init_grad
                else:
                    assert (
                        line_search_fn == "strong_wolfe"
                    ), "only strong_wolfe is expected"
                    init_param = np.copy(x)

                    def eval_func(x, t, d):
                        return (
                            compute_loss(x + t * d, init_grad),
                            2 * (x + t * d) * init_grad,
                        )

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        eval_func, init_param, t, d, loss, flat_grad, gtd
                    )
                    x += t * d
                current_evals += ls_func_evals
                state["func_evals"] += ls_func_evals
                if n_iter == max_iter:
                    break

                if current_evals >= max_eval:
                    break

                if np.max(np.abs(flat_grad)) <= tolerance_grad:
                    break

                if np.max(np.abs(d * t)) <= tolerance_change:
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
            return x

        state = defaultdict(dict)
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)
        for _ in range(0, train_iters):
            x = np_train_one_iter(x, state, np.concatenate(random_grad_seq))
        return x

    oneflow_res = flow.cat(train_by_oneflow(), 0)
    numpy_res = train_by_numpy()
    test_case.assertTrue(
        np.allclose(
            oneflow_res.numpy().flatten(), numpy_res.flatten(), rtol=0.01, atol=0.01,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestLBFGS(flow.unittest.TestCase):
    def test_lbfgs_numpy(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = [random_device().value()]
        arg_dict["x_shape"] = [10, 20]
        arg_dict["learning_rate"] = [0.01]
        arg_dict["train_iters"] = [20]
        arg_dict["max_iter"] = [20]
        arg_dict["max_eval"] = [25]
        arg_dict["tolerance_grad"] = [1e-7]
        arg_dict["tolerance_change"] = [1e-9]
        arg_dict["history_size"] = [100]
        arg_dict["line_search_fn"] = [None, "strong_wolfe"]
        arg_dict["reload_state_step"] = [5]
        arg_dict["save_load_by_pickle"] = [random_bool().value()]
        arg_dict["contiguous_params"] = [random_bool().value()]
        arg_dict["tensor_num"] = [3, 4, 7]
        arg_dict["use_float64"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_lbfgs(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
