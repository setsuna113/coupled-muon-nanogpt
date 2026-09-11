"""TangentMuon: orthogonalize the product-space momentum of a weight pair.

Port of ``muon_bench/tangent_muon.py`` (setsuna113/Muon; derivation in
``docs/tangent_muon_derivation.pdf``) from LoRA ``(B, A)`` factors to this
harness's coupled pairs: Q-K and V-O per attention head, MLP up-down, the
MLA K-side pair and the imposed-FactFF pair.

Setting. A pair ``(P, Q)`` with ``P: (out, k)`` and ``Q: (k, in)`` enters the
forward pass only through the product ``P @ Q`` (``Q_h^T K_h`` for one
attention head, ``O_h V_h``, ``down @ up``). With heavy-ball momenta
``M_P, M_Q`` the first-order change of the product is

    C = P M_Q + M_P Q                       (out, in), rank <= 2k.

``CoupledMuon_v2`` orthogonalizes one factor at a time given the partner's
current value. TangentMuon orthogonalizes ``C`` itself and splits the result
back to the factors:

  v3   Z = msign(C);  solve  (P P^T) L + L (Q^T Q) + lam L = C - Z;
       U_P = M_P - L Q^T,   U_Q = M_Q - P^T L      =>   P U_Q + U_P Q = Z + lam L.
  v7   solve  (P P^T) L0 + L0 (Q^T Q) + lam L0 = C;   L = msign(L0);
       U_P = L Q^T,   U_Q = P^T L.

The damped Sylvester equation is solved through the eigendecompositions of
the two Grams. Because ``C`` has rank <= 2k, whenever ``2k < min(out, in)``
the whole computation runs on a ``2k x 2k`` core obtained from compact QRs of
``[P | M_P]`` and ``[M_Q ; Q]``; otherwise the dense ``(out, in)`` form is
used. The two routes are algebraically identical (``tests/test_tangent_muon.py``).

The joint update is rescaled so that ``||(U_P, U_Q)||_F^2 = min(out,k) + min(k,in)``
(one Muon-sized step per factor) and written back with the same per-matrix LR
prefactor as ``Muon`` / ``CoupledMuon_v2``. All pair math runs in fp32; the
plain-Muon and AdamW paths for unpaired parameters are the same as in
``CoupledMuon_v2``.

Per-head geometry. For Q-K the pair for head ``h`` is ``(Q_h^T, K_h)`` with
``Q_h, K_h: (head_dim, hidden)``; for V-O it is ``(O_h, V_h)`` with
``O_h: (hidden, head_dim)`` and ``V_h: (head_dim, hidden)``. Under GQA every
query head is paired with its shared KV head and the KV update is averaged
over the group. No RoPE-specific split is applied (tangent.pdf V3); the
per-head structure alone is what the derivation needs.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .coupled_muon import zeropower_via_newtonschulz5

VARIANTS = ("v3", "v7")
ROUTES = ("auto", "lowrank", "dense")


def damping_lambda(P: Tensor, Q: Tensor, k: int, damping: float) -> Tensor:
    """lam = damping * mean diagonal of the two Grams in 2k-dimensional
    coordinates, i.e. ``damping * (tr(P P^T) + tr(Q^T Q)) / (2 * 2k)``.
    Defined from the pair itself so the low-rank and dense routes agree."""
    return float(damping) * 0.5 * ((P * P).sum() + (Q * Q).sum()) / float(2 * k)


def solve_damped_sylvester(G_L: Tensor, G_R: Tensor, rhs: Tensor, lam: Tensor | float) -> Tensor:
    """Solve ``G_L X + X G_R + lam X = rhs`` for symmetric PSD ``G_L (m,m)``,
    ``G_R (n,n)`` and ``rhs (m,n)`` by diagonalizing both Grams."""
    D_L, S_L = torch.linalg.eigh(G_L)
    D_R, S_R = torch.linalg.eigh(G_R)
    rhs_e = S_L.T @ rhs @ S_R
    denom = (D_L.unsqueeze(1) + D_R.unsqueeze(0) + lam).clamp(min=1e-12)
    return S_L @ (rhs_e / denom) @ S_R.T


def _msign(X: Tensor, steps: int, coeffs: Tensor | None) -> Tensor:
    return zeropower_via_newtonschulz5(X, steps=steps, dtype=torch.float32, coeffs=coeffs).to(torch.float32)


def tangent_pair_update(
    P: Tensor,
    Q: Tensor,
    M_P: Tensor,
    M_Q: Tensor,
    *,
    variant: str = "v3",
    ns_steps: int = 5,
    damping: float = 0.1,
    coeffs: Tensor | None = None,
    route: str = "auto",
) -> tuple[Tensor, Tensor]:
    """One TangentMuon update for the pair ``(P, Q)`` with momenta ``(M_P, M_Q)``.

    Returns ``(U_P, U_Q)`` in fp32, jointly normalized to Muon scale. The
    caller applies the per-matrix LR prefactor and weight decay.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    P = P.to(torch.float32)
    Q = Q.to(torch.float32)
    M_P = M_P.to(torch.float32)
    M_Q = M_Q.to(torch.float32)
    out, k = P.shape
    k2, inn = Q.shape
    if k != k2 or M_P.shape != P.shape or M_Q.shape != Q.shape:
        raise ValueError(
            f"shape mismatch: P{tuple(P.shape)} Q{tuple(Q.shape)} "
            f"M_P{tuple(M_P.shape)} M_Q{tuple(M_Q.shape)}"
        )
    if route == "auto":
        route = "lowrank" if 2 * k < min(out, inn) else "dense"
    lam = damping_lambda(P, Q, k, damping)

    if route == "lowrank":
        # C = [P | M_P] @ [M_Q ; Q]  =  Q_L (R_L R_R^T) Q_R^T
        L = torch.cat([P, M_P], dim=1)  # (out, 2k)
        R = torch.cat([M_Q, Q], dim=0)  # (2k, in)
        Q_L, R_L = torch.linalg.qr(L)  # (out, 2k), (2k, 2k)
        Q_R, R_R = torch.linalg.qr(R.T)  # (in, 2k), (2k, 2k)
        core = R_L @ R_R.T  # C in the (Q_L, Q_R) coordinates
        Phi = R_L[:, :k]  # P   = Q_L Phi
        Psi = R_R[:, k:]  # Q^T = Q_R Psi
        G_L = Phi @ Phi.T
        G_R = Psi @ Psi.T
        if variant == "v3":
            Zc = _msign(core, ns_steps, coeffs)
            Lam = solve_damped_sylvester(G_L, G_R, core - Zc, lam)
            U_P = M_P - Q_L @ Lam @ Psi
            U_Q = M_Q - Phi.T @ Lam @ Q_R.T
        else:
            Lam0 = solve_damped_sylvester(G_L, G_R, core, lam)
            Lam = _msign(Lam0, ns_steps, coeffs)
            U_P = Q_L @ Lam @ Psi
            U_Q = Phi.T @ Lam @ Q_R.T
    else:
        C = P @ M_Q + M_P @ Q  # (out, in)
        G_L = P @ P.T  # (out, out)
        G_R = Q.T @ Q  # (in, in)
        if variant == "v3":
            Z = _msign(C, ns_steps, coeffs)
            Lam = solve_damped_sylvester(G_L, G_R, C - Z, lam)
            U_P = M_P - Lam @ Q.T
            U_Q = M_Q - P.T @ Lam
        else:
            Lam0 = solve_damped_sylvester(G_L, G_R, C, lam)
            Lam = _msign(Lam0, ns_steps, coeffs)
            U_P = Lam @ Q.T
            U_Q = P.T @ Lam

    # Joint scalar normalization (tangent.pdf section 5): one Muon-sized step
    # per factor, shared scalar so the product-space direction is preserved.
    target = float(min(U_P.shape) + min(U_Q.shape))
    norm = torch.sqrt((U_P * U_P).sum() + (U_Q * U_Q).sum())
    s = math.sqrt(target) / (norm + 1e-7)
    return U_P * s, U_Q * s


class TangentMuon(torch.optim.Optimizer):
    """TangentMuon for coupled pairs + plain Muon for other matrices + AdamW backup.

    Arguments mirror ``CoupledMuon_v2`` so the factory can build either from
    the same pair classification:
        coupled_pairs: list of ``(param_A, param_B, n_heads_for_pair, is_qk)``.
            The product is ``A @ B`` for every kind except ``qk``, where it is
            ``A^T @ B`` (``Q^T K``) per head.
        pair_kinds: optional list aligned with ``coupled_pairs`` with values
            ``"qk" | "vo" | "flat"``. Defaults to ``"qk"`` for ``is_qk`` pairs
            and ``"flat"`` otherwise.
        tangent_variant: ``"v3"`` (balanced Sylvester split, default) or
            ``"v7"`` (matrix sign in coefficient space).
        damping: Sylvester regulariser relative to the pair's Gram magnitude.
        use_multi_head: per-head treatment of the ``qk`` / ``vo`` pairs.
        ns_steps: Newton-Schulz steps for the core / product and for plain Muon.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.1,
        coupled_pairs=None,
        pair_kinds=None,
        muon_params=None,
        adamw_params=None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas=(0.95, 0.95),
        adamw_eps: float = 1e-8,
        tangent_variant: str = "v3",
        damping: float = 0.1,
        use_multi_head: bool = True,
        n_heads: int = 1,
        ns_dtype: torch.dtype = torch.bfloat16,
        ns_coefficients: str = "bernstein",
        lr_prefactor: str = "moonlight",
    ):
        coupled_pairs = list(coupled_pairs or [])
        muon_params = list(muon_params or [])
        adamw_params = list(adamw_params or [])
        if pair_kinds is None:
            pair_kinds = ["qk" if is_qk else "flat" for (_a, _b, _h, is_qk) in coupled_pairs]
        pair_kinds = [str(kd) for kd in pair_kinds]
        if len(pair_kinds) != len(coupled_pairs):
            raise ValueError("pair_kinds must align with coupled_pairs")
        for kd in pair_kinds:
            if kd not in ("qk", "vo", "flat"):
                raise ValueError(f"pair kind must be qk | vo | flat, got {kd!r}")
        if tangent_variant not in VARIANTS:
            raise ValueError(f"tangent_variant must be one of {VARIANTS}, got {tangent_variant!r}")

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        all_params = []
        for param_A, param_B, _h, _is_qk in coupled_pairs:
            all_params.extend([param_A, param_B])
        all_params.extend(muon_params)
        all_params.extend(adamw_params)
        super().__init__(all_params, defaults)

        self.coupled_pairs = coupled_pairs
        self._pair_kinds = pair_kinds
        self.tangent_variant = tangent_variant
        self.damping = float(damping)
        self.use_multi_head = bool(use_multi_head)
        self.n_heads = int(n_heads)
        self.ns_dtype = ns_dtype
        self.lr_prefactor = str(lr_prefactor).lower()
        self._ns_steps = int(ns_steps)

        from .ns_coefficients import coefficients_to_tensor, get_coefficients

        self.ns_coefficients_policy = str(ns_coefficients).lower()
        if self.ns_coefficients_policy == "bernstein":
            self._coeffs = None
        else:
            self._coeffs = coefficients_to_tensor(
                get_coefficients(self.ns_coefficients_policy, self._ns_steps)
            )

        for param_A, param_B, n_heads_pair, is_qk in coupled_pairs:
            assert param_A.ndim == 2 and param_B.ndim == 2
            self.state[param_A]["use_coupled"] = True
            self.state[param_A]["coupled_with"] = param_B
            self.state[param_A]["is_A"] = True
            self.state[param_A]["is_qk"] = is_qk
            self.state[param_A]["num_heads"] = n_heads_pair
            self.state[param_B]["use_coupled"] = True
            self.state[param_B]["coupled_with"] = param_A
            self.state[param_B]["is_A"] = False
            self.state[param_B]["is_qk"] = is_qk
            self.state[param_B]["num_heads"] = n_heads_pair
        for p in muon_params:
            assert p.ndim == 2
            self.state[p]["use_muon"] = True
            self.state[p]["use_coupled"] = False
        for p in adamw_params:
            self.state[p]["use_muon"] = False
            self.state[p]["use_coupled"] = False

    # ------------------------------------------------------------------
    def adjust_lr_for_muon(self, lr: float, param_shape) -> float:
        """Same per-matrix LR prefactor menu as ``Muon`` / ``CoupledMuon_v2``."""
        A, B = param_shape[:2]
        policy = self.lr_prefactor
        if policy == "moonlight":
            return lr * 0.2 * math.sqrt(max(A, B))
        if policy == "bernstein_ratio":
            return lr * 0.2 * math.sqrt(A / max(B, 1))
        if policy == "cesista":
            K = max(self._ns_steps, 1)
            return lr * 0.2 * math.sqrt(max(A, B)) / (1.0 + math.log(K + 1))
        raise ValueError(
            f"Unknown lr_prefactor={policy!r}. Supported: moonlight | bernstein_ratio | cesista."
        )

    def _pair_update(self, P, Q, M_P, M_Q):
        return tangent_pair_update(
            P, Q, M_P, M_Q,
            variant=self.tangent_variant, ns_steps=self._ns_steps,
            damping=self.damping, coeffs=self._coeffs, route="auto",
        )

    # ---- attention pairs ------------------------------------------------
    def _qk_update(self, q: Tensor, k: Tensor, m_q: Tensor, m_k: Tensor, n_heads: int):
        """Q-K: ``q, k: (n_heads*d_h, hidden)``; per head the pair is ``(Q_h^T, K_h)``."""
        HDq, I = q.shape
        HDk, _ = k.shape
        if not self.use_multi_head and HDq == HDk:
            U_P, U_Q = self._pair_update(q.T, k, m_q.T, m_k)
            return U_P.T, U_Q
        d_h = HDq // n_heads
        nq = n_heads
        nk = HDk // d_h
        n_rep = nq // nk
        q3, k3 = q.view(nq, d_h, I), k.view(nk, d_h, I)
        mq3, mk3 = m_q.view(nq, d_h, I), m_k.view(nk, d_h, I)
        U_q = torch.zeros(nq, d_h, I, dtype=torch.float32, device=q.device)
        U_k = torch.zeros(nk, d_h, I, dtype=torch.float32, device=q.device)
        for h in range(nq):
            j = h // n_rep
            U_P, U_Q = self._pair_update(q3[h].T, k3[j], mq3[h].T, mk3[j])
            U_q[h] = U_P.T
            U_k[j] += U_Q / n_rep
        return U_q.reshape(HDq, I), U_k.reshape(HDk, I)

    def _vo_update(self, o: Tensor, v: Tensor, m_o: Tensor, m_v: Tensor, n_kv: int):
        """V-O: ``o: (hidden, n_heads*d_h)``, ``v: (n_kv*d_h, hidden)``; per head ``(O_h, V_h)``."""
        I, HDo = o.shape
        HDv, _ = v.shape
        if not self.use_multi_head and HDo == HDv:
            return self._pair_update(o, v, m_o, m_v)
        d_h = HDv // n_kv
        nv = n_kv
        no = HDo // d_h
        n_rep = no // nv
        o3 = o.view(I, no, d_h).permute(1, 0, 2)
        mo3 = m_o.view(I, no, d_h).permute(1, 0, 2)
        v3, mv3 = v.view(nv, d_h, I), m_v.view(nv, d_h, I)
        U_o = torch.zeros(no, I, d_h, dtype=torch.float32, device=o.device)
        U_v = torch.zeros(nv, d_h, I, dtype=torch.float32, device=o.device)
        for h in range(no):
            j = h // n_rep
            U_P, U_Q = self._pair_update(o3[h], v3[j], mo3[h], mv3[j])
            U_o[h] = U_P
            U_v[j] += U_Q / n_rep
        return U_o.permute(1, 0, 2).reshape(I, HDo), U_v.reshape(HDv, I)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            # ---- coupled pairs --------------------------------------------
            processed = set()
            for (param_A, param_B, n_heads_pair, _is_qk), kind in zip(
                self.coupled_pairs, self._pair_kinds, strict=True
            ):
                if param_A in processed or param_B in processed:
                    continue
                g_A, g_B = param_A.grad, param_B.grad
                if g_A is None or g_B is None:
                    continue
                state_A, state_B = self.state[param_A], self.state[param_B]
                if "momentum_buffer" not in state_A:
                    state_A["momentum_buffer"] = torch.zeros_like(g_A)
                if "momentum_buffer" not in state_B:
                    state_B["momentum_buffer"] = torch.zeros_like(g_B)
                buf_A, buf_B = state_A["momentum_buffer"], state_B["momentum_buffer"]
                buf_A.mul_(momentum).add_(g_A)
                buf_B.mul_(momentum).add_(g_B)
                if group["nesterov"]:
                    g_A_eff = g_A.add(buf_A, alpha=momentum)
                    g_B_eff = g_B.add(buf_B, alpha=momentum)
                else:
                    g_A_eff, g_B_eff = buf_A, buf_B

                A32 = param_A.data.to(torch.float32)
                B32 = param_B.data.to(torch.float32)
                mA32 = g_A_eff.to(torch.float32)
                mB32 = g_B_eff.to(torch.float32)
                if kind == "qk":
                    u_A, u_B = self._qk_update(A32, B32, mA32, mB32, int(n_heads_pair))
                elif kind == "vo":
                    u_A, u_B = self._vo_update(A32, B32, mA32, mB32, int(n_heads_pair))
                else:
                    u_A, u_B = self._pair_update(A32, B32, mA32, mB32)

                adjusted_lr_A = self.adjust_lr_for_muon(lr, param_A.shape)
                adjusted_lr_B = self.adjust_lr_for_muon(lr, param_B.shape)
                param_A.data.mul_(1 - lr * wd).add_(u_A.to(param_A.dtype), alpha=-adjusted_lr_A)
                param_B.data.mul_(1 - lr * wd).add_(u_B.to(param_B.dtype), alpha=-adjusted_lr_B)
                processed.add(param_A)
                processed.add(param_B)

            # ---- plain Muon -----------------------------------------------
            for p in (q for q in group["params"] if self.state[q].get("use_muon", False)):
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(g, steps=ns_steps, dtype=self.ns_dtype, coeffs=self._coeffs)
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd).add_(u, alpha=-adjusted_lr)

            # ---- AdamW backup ---------------------------------------------
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for p in (
                q for q in group["params"]
                if not self.state[q].get("use_coupled", False) and not self.state[q].get("use_muon", False)
            ):
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1, buf2 = state["moment1"], state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g_hat = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * wd).add_(g_hat, alpha=-lr / scale)

        return loss
