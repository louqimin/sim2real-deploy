# AGIself · D1 edu 部署侧策略推理
"""用 onnxruntime 加载导出的 policy.onnx，45 维进、12 维出。

为什么是 ONNX 而不是 TorchScript：加载 .pt 必须装 torch，而板载计算机能不能装 torch
至今是未收的钉子（D12）。ONNX 只要 onnxruntime，PyPI 上 x86 与 aarch64 都有现成轮子，
体积小一个数量级。这个策略就是个小 MLP，50 Hz 下用哪个运行时都绰绰有余 ——
换句话说，规避 D12 这条风险的代价是零。

policy.pt 不扔，留在 bundle 里当对账用的标准答案（见 tools/AGIself_replay_check.py）。
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from .contract import Contract, get_contract


class OnnxPolicy:
    """ONNX 策略网络的薄封装。加载时就把输入输出签名与契约对一遍。"""

    def __init__(
        self,
        policy_path: Optional[str] = None,
        contract: Optional[Contract] = None,
        providers: Optional[Sequence[str]] = None,
    ):
        import onnxruntime as ort  # 延迟 import：contract.py 不该被 onnxruntime 绑架

        self.c = contract if contract is not None else get_contract()
        self.path = policy_path or self.c.policy_path

        opts = ort.SessionOptions()
        # 固定单线程。50 Hz 下这个 MLP 根本不需要并行，而线程池在实时控制循环里
        # 只会带来抖动 —— 抖动是本项目的硬件红线之一，能少一个来源就少一个。
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        self.sess = ort.InferenceSession(
            self.path,
            sess_options=opts,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError(
                f"期望单进单出的策略网络，实际 {len(ins)} 个输入 / {len(outs)} 个输出。"
                f"输入 {[i.name for i in ins]}，输出 {[o.name for o in outs]}"
            )
        self.input_name = ins[0].name
        self.output_name = outs[0].name
        self.input_shape = list(ins[0].shape)
        self.output_shape = list(outs[0].shape)

        # 形状里可能有 'batch' 之类的符号维（导出时标成动态轴），只校验最后一维。
        last_in = self.input_shape[-1] if self.input_shape else None
        if isinstance(last_in, int) and last_in != self.c.obs_dim:
            raise ValueError(
                f"policy.onnx 的输入最后一维是 {last_in}，契约说观测是 {self.c.obs_dim} 维。"
                f"policy 与契约不配套 —— 重跑 AGIself_make_bundle.py 把两者一起换掉。"
            )
        last_out = self.output_shape[-1] if self.output_shape else None
        if isinstance(last_out, int) and last_out != self.c.action_dim:
            raise ValueError(
                f"policy.onnx 的输出最后一维是 {last_out}，契约说动作是 {self.c.action_dim} 维"
            )

        # 输入是 (45,) 还是 (1, 45)，看导出时的秩，照着喂。
        self._rank = len(self.input_shape) if self.input_shape else 1
        self._feed_shape = (1, self.c.obs_dim) if self._rank >= 2 else (self.c.obs_dim,)

    # ---------- 推理 ----------

    def __call__(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(self._feed_shape)
        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        a = np.asarray(y, dtype=np.float32).reshape(-1)
        if a.size != self.c.action_dim:
            raise RuntimeError(f"策略输出了 {a.size} 个数，契约要求 {self.c.action_dim} 个")
        return a

    # ---------- 自检 ----------

    def warmup(self, n: int = 20) -> None:
        """先空跑几帧。第一次 run 会做内存分配与图优化，比稳态慢一个量级；
        不预热的话，控制循环的第一帧会是个突出的延迟尖峰。"""
        z = np.zeros(self.c.obs_dim, dtype=np.float32)
        for _ in range(n):
            self(z)

    def benchmark(self, n: int = 500) -> dict:
        """量一下单帧推理耗时，与 control_dt 比。

        判据很直白：均值必须远小于 control_dt（20 ms），否则 50 Hz 根本跑不住。
        """
        self.warmup()
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((n, self.c.obs_dim)).astype(np.float32) * 0.1
        t = np.empty(n, dtype=np.float64)
        for i in range(n):
            t0 = time.perf_counter()
            self(samples[i])
            t[i] = (time.perf_counter() - t0) * 1e3
        return {
            "mean_ms": float(t.mean()),
            "p50_ms": float(np.percentile(t, 50)),
            "p99_ms": float(np.percentile(t, 99)),
            "max_ms": float(t.max()),
            "control_dt_ms": self.c.control_dt * 1e3,
            "headroom_x": float(self.c.control_dt * 1e3 / max(t.mean(), 1e-9)),
        }
