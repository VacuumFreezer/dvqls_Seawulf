"""Qiskit/Aer MPS builder for the 2x2 cluster30 distributed VQLS objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2

from .circuits_cat_nodispatch_qiskit import (
    CircuitTemplate,
    aggregate_pauli_operator,
    build_beta_template,
    build_expectation_template,
    build_overlap_template,
    make_term_bundle_qiskit,
    pauli_word_to_observable,
)
from .static_ops_2x2_cluster30_qiskit import BStateSpec, LinearSystemDataQiskit, PauliWord


@dataclass
class EntrySpec:
    sys_id: int
    agent_id: int
    neighbors: Tuple[int, ...]
    degree: int
    coeffs: np.ndarray
    c_vec: np.ndarray
    words: Tuple[PauliWord, ...]
    b_spec: BStateSpec
    terms: object


def _template_signature(template: CircuitTemplate) -> Tuple[str, str, Tuple[str, ...]]:
    param_names = tuple(p.name for p in template.ordered_parameters)
    block_names = tuple(sorted(template.blocks.keys()))
    return (
        template.name,
        f"nq={template.circuit.num_qubits}|obs={len(template.observables)}",
        block_names + param_names,
    )


class DistributedCostBuilderQiskit:
    def __init__(
        self,
        system: LinearSystemDataQiskit,
        row_topology: Mapping[int, Sequence[int]],
        *,
        ansatz_layers: int = 1,
        repeat_cz_each_layer: bool = False,
        ansatz_kind: str | None = None,
        scaffold_edges: Sequence[Tuple[int, int]] | None = None,
        max_bond_dim: int = 8,
        num_threads: int = 1,
        precision: str = "single",
        seed: int = 0,
        optimization_level: int = 0,
    ):
        self.system = system
        self.row_topology = {int(k): tuple(int(x) for x in v) for k, v in row_topology.items()}
        self.n_input_qubit = int(system.n_data_qubits)
        self.ansatz_layers = int(ansatz_layers)
        self.repeat_cz_each_layer = bool(repeat_cz_each_layer)
        metadata = getattr(system, "metadata", {}) or {}
        self.ansatz_kind = str(
            ansatz_kind
            or metadata.get("recommended_ansatz_qiskit")
            or metadata.get("recommended_ansatz")
            or "cluster_h_cz_ry"
        )
        if scaffold_edges is None:
            scaffold_edges = metadata.get("cluster_scaffold_edges_local")
        self.scaffold_edges = None if scaffold_edges is None else tuple((int(a), int(b)) for a, b in scaffold_edges)
        self.num_threads = int(num_threads)
        self.max_bond_dim = int(max_bond_dim)
        self.precision = str(precision)
        self.seed = int(seed)
        self.optimization_level = int(optimization_level)

        backend_options = {
            "device": "CPU",
            "method": "matrix_product_state",
            "precision": self.precision,
            "matrix_product_state_max_bond_dimension": self.max_bond_dim,
            "mps_omp_threads": self.num_threads,
            "runtime_parameter_bind_enable": True,
            "fusion_enable": False,
            "max_parallel_experiments": 1,
            "seed_simulator": self.seed,
        }
        self.backend = AerSimulator(**backend_options)
        self.estimator = EstimatorV2(
            options={
                "default_precision": 0.0,
                "backend_options": backend_options,
            }
        )

        self._template_cache: Dict[Tuple[str, str, Tuple[str, ...]], CircuitTemplate] = {}
        self._word_overlap_cache: Dict[str, CircuitTemplate] = {}
        self._bprep_overlap_cache: Dict[Tuple[str, str], CircuitTemplate] = {}
        self._direct_operator_cache: Dict[Tuple[str, Tuple[str, ...], Tuple[float, ...]], CircuitTemplate] = {}

        self.entries = self._build_entries()
        self._diag_words = tuple(self.system.gates_grid[0][0])
        self._diag_coeffs = np.asarray(self.system.coeffs[0][0], dtype=np.float32)
        self._off_words = tuple(self.system.gates_grid[0][1])
        self._off_coeffs = np.asarray(self.system.coeffs[0][1], dtype=np.float32)
        self._res_self_words, self._res_self_coeffs = self._build_residual_self_operator()
        self._res_cross_words, self._res_cross_coeffs = aggregate_pauli_operator(
            self._diag_words,
            self._diag_coeffs,
            self._off_words,
            self._off_coeffs,
        )
        self._res_self_template = self._get_direct_operator_template(
            "res_self",
            self._res_self_words,
            self._res_self_coeffs,
        )

    def _transpile_template(self, template: CircuitTemplate) -> CircuitTemplate:
        signature = _template_signature(template)
        if signature in self._template_cache:
            return self._template_cache[signature]
        self._template_cache[signature] = template
        return template

    def _get_word_overlap_template(self, word: PauliWord | None) -> CircuitTemplate:
        key = "I" if word is None else word.label
        if key not in self._word_overlap_cache:
            tmpl = build_overlap_template(
                n_data_qubits=self.n_input_qubit,
                layers=self.ansatz_layers,
                repeat_cz_each_layer=self.repeat_cz_each_layer,
                ansatz_kind=self.ansatz_kind,
                scaffold_edges=self.scaffold_edges,
                left_kind="ansatz",
                right_kind="ansatz",
                pauli_word=word,
                left_name="left",
                right_name="right",
                template_name=f"alpha_alpha_{key}",
            )
            self._word_overlap_cache[key] = self._transpile_template(tmpl)
        return self._word_overlap_cache[key]

    def _get_bprep_overlap_template(self, b_spec: BStateSpec, word: PauliWord | None) -> CircuitTemplate:
        key = (b_spec.label, "I" if word is None else word.label)
        if key not in self._bprep_overlap_cache:
            tmpl = build_overlap_template(
                n_data_qubits=self.n_input_qubit,
                layers=self.ansatz_layers,
                repeat_cz_each_layer=self.repeat_cz_each_layer,
                ansatz_kind=self.ansatz_kind,
                scaffold_edges=self.scaffold_edges,
                left_kind="bprep",
                right_kind="ansatz",
                pauli_word=word,
                left_bspec=b_spec,
                right_name="theta",
                template_name=f"bprep_{b_spec.label}_{key[1]}",
            )
            self._bprep_overlap_cache[key] = self._transpile_template(tmpl)
        return self._bprep_overlap_cache[key]

    def _get_direct_operator_template(
        self,
        name: str,
        words: Sequence[PauliWord],
        coeffs: Sequence[float],
    ) -> CircuitTemplate:
        key = (str(name), tuple(word.label for word in words), tuple(float(c) for c in coeffs))
        if key not in self._direct_operator_cache:
            observables = [
                pauli_word_to_observable(word, self.n_input_qubit, coeff=float(coeff))
                for word, coeff in zip(words, coeffs)
            ]
            tmpl = build_expectation_template(
                n_data_qubits=self.n_input_qubit,
                observables=observables,
                layers=self.ansatz_layers,
                repeat_cz_each_layer=self.repeat_cz_each_layer,
                ansatz_kind=self.ansatz_kind,
                scaffold_edges=self.scaffold_edges,
                theta_name="theta",
                template_name=str(name),
            )
            self._direct_operator_cache[key] = self._transpile_template(tmpl)
        return self._direct_operator_cache[key]

    def _build_entries(self) -> List[EntrySpec]:
        entries: List[EntrySpec] = []
        for sys_id in range(int(self.system.n)):
            for agent_id in range(int(self.system.n)):
                words = tuple(self.system.gates_grid[sys_id][agent_id])
                coeffs = np.asarray(self.system.coeffs[sys_id][agent_id], dtype=np.float32)
                b_spec = self.system.b_specs[sys_id][agent_id]
                term_bundle = make_term_bundle_qiskit(
                    n_input_qubit=self.n_input_qubit,
                    U_spec=b_spec,
                    A_words=words,
                    coeffs=coeffs,
                    layers=self.ansatz_layers,
                    repeat_cz_each_layer=self.repeat_cz_each_layer,
                    ansatz_kind=self.ansatz_kind,
                    scaffold_edges=self.scaffold_edges,
                )
                term_bundle.omega = self._transpile_template(term_bundle.omega)
                term_bundle.delta = self._transpile_template(term_bundle.delta)
                term_bundle.beta = self._transpile_template(term_bundle.beta)
                term_bundle.zeta = {
                    label: self._transpile_template(tmpl) for label, tmpl in term_bundle.zeta.items()
                }
                term_bundle.tau = {
                    label: self._transpile_template(tmpl) for label, tmpl in term_bundle.tau.items()
                }

                neighbors = self.row_topology[agent_id]
                degree = len(neighbors)
                c_vec = np.asarray([-float(degree)] + [1.0] * degree, dtype=np.float32)
                entries.append(
                    EntrySpec(
                        sys_id=sys_id,
                        agent_id=agent_id,
                        neighbors=neighbors,
                        degree=degree,
                        coeffs=coeffs,
                        c_vec=c_vec,
                        words=words,
                        b_spec=b_spec,
                        terms=term_bundle,
                    )
                )
        return entries

    def _build_residual_self_operator(self) -> Tuple[List[PauliWord], np.ndarray]:
        d2_words, d2_coeffs = aggregate_pauli_operator(self._diag_words, self._diag_coeffs)
        o2_words, o2_coeffs = aggregate_pauli_operator(self._off_words, self._off_coeffs)
        acc: Dict[Tuple[Tuple[int, str], ...], float] = {}
        labels: Dict[Tuple[Tuple[int, str], ...], str] = {}
        for word, coeff in list(zip(d2_words, d2_coeffs)) + list(zip(o2_words, o2_coeffs)):
            acc[word.ops] = acc.get(word.ops, 0.0) + float(coeff)
            labels[word.ops] = word.label
        words = [PauliWord(key, labels[key]) for key in acc]
        coeffs = np.asarray([acc[key] for key in acc], dtype=np.float32)
        return words, coeffs

    def _build_agent_view(self, entry: EntrySpec, params: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        sys_id = entry.sys_id
        agent_id = entry.agent_id
        betas = [params["beta"][sys_id, agent_id]]
        lams = [params["lambda"][sys_id, agent_id]]
        for nbr in entry.neighbors:
            betas.append(params["beta"][sys_id, nbr])
            lams.append(params["lambda"][sys_id, nbr])
        return {
            "alpha": params["alpha"][sys_id, agent_id],
            "beta_vec": np.asarray(betas, dtype=np.float32),
            "lam_vec": np.asarray(lams, dtype=np.float32),
            "sigma": np.float32(params["sigma"][sys_id, agent_id]),
            "b_norm": np.float32(params["b_norm"][sys_id, agent_id]),
        }

    def _evaluate_local_term_cache(self, params: Mapping[str, np.ndarray]) -> List[Dict[str, object]]:
        pubs = []
        layouts = []
        locals_cache = []

        for entry_idx, entry in enumerate(self.entries):
            local = self._build_agent_view(entry, params)
            locals_cache.append(local)
            alpha = local["alpha"]
            beta_vec = local["beta_vec"]

            tau_count = len(entry.words)
            for word in entry.words:
                pubs.append(
                    (
                        entry.terms.tau[word.label].circuit,
                        entry.terms.tau[word.label].observable_arg,
                        [entry.terms.tau[word.label].pack(alpha=alpha)],
                    )
                )
            tau_start = len(pubs) - tau_count

            pubs.append(
                (
                    entry.terms.beta.circuit,
                    entry.terms.beta.observable_arg,
                    [entry.terms.beta.pack(alpha=alpha)],
                )
            )
            beta_idx = len(pubs) - 1

            zeta_ranges = []
            for beta in beta_vec:
                start = len(pubs)
                for word in entry.words:
                    tmpl = entry.terms.zeta[word.label]
                    pubs.append((tmpl.circuit, tmpl.observable_arg, [tmpl.pack(alpha=alpha, beta=beta)]))
                zeta_ranges.append((start, len(entry.words)))

            delta_indices = []
            for beta in beta_vec:
                pubs.append(
                    (
                        entry.terms.delta.circuit,
                        entry.terms.delta.observable_arg,
                        [entry.terms.delta.pack(beta=beta)],
                    )
                )
                delta_indices.append(len(pubs) - 1)

            omega_pairs = []
            for i in range(len(beta_vec)):
                for j in range(i + 1, len(beta_vec)):
                    pubs.append(
                        (
                            entry.terms.omega.circuit,
                            entry.terms.omega.observable_arg,
                            [entry.terms.omega.pack(left=beta_vec[i], right=beta_vec[j])],
                        )
                    )
                    omega_pairs.append((i, j, len(pubs) - 1))

            layouts.append(
                {
                    "entry_idx": entry_idx,
                    "tau_start": tau_start,
                    "tau_count": tau_count,
                    "beta_idx": beta_idx,
                    "zeta_ranges": zeta_ranges,
                    "delta_indices": delta_indices,
                    "omega_pairs": omega_pairs,
                }
            )

        result = self.estimator.run(pubs).result()
        pub_values: List[np.ndarray] = [np.asarray(pub_res.data.evs, dtype=np.float64).reshape(-1) for pub_res in result]

        records: List[Dict[str, object]] = []
        for layout, local in zip(layouts, locals_cache):
            entry = self.entries[layout["entry_idx"]]
            beta_vec = local["beta_vec"]
            lam_vec = np.asarray(local["lam_vec"], dtype=np.float64)
            c_vec = np.asarray(entry.c_vec, dtype=np.float64)
            sigma = float(local["sigma"])
            b_norm = float(local["b_norm"])

            tau_re = 0.0
            for offset, idx in enumerate(range(layout["tau_start"], layout["tau_start"] + layout["tau_count"])):
                tau_re += float(entry.coeffs[offset]) * float(pub_values[idx][0])

            beta_re = float(np.sum(pub_values[layout["beta_idx"]]))

            zeta_vec = np.zeros((len(beta_vec),), dtype=np.float64)
            for k, (start, count) in enumerate(layout["zeta_ranges"]):
                zeta_vec[k] = float(
                    sum(float(entry.coeffs[offset]) * float(pub_values[idx][0]) for offset, idx in enumerate(range(start, start + count)))
                )

            delta_re_vec = np.asarray([float(pub_values[idx][0]) for idx in layout["delta_indices"]], dtype=np.float64)

            omega_mat = np.zeros((len(beta_vec), len(beta_vec)), dtype=np.float64)
            for i, j, idx in layout["omega_pairs"]:
                omega_mat[i, j] = float(pub_values[idx][0])
                omega_mat[j, i] = omega_mat[i, j]

            records.append(
                {
                    "entry": entry,
                    "lam_vec": lam_vec,
                    "c_vec": c_vec,
                    "sigma": sigma,
                    "b_norm": b_norm,
                    "tau_re": tau_re,
                    "beta_re": beta_re,
                    "zeta_vec": zeta_vec,
                    "delta_re_vec": delta_re_vec,
                    "omega_mat": omega_mat,
                }
            )

        return records

    @staticmethod
    def _loss_from_record(record: Mapping[str, object]) -> tuple[float, np.ndarray]:
        lam_vec = np.asarray(record["lam_vec"], dtype=np.float64)
        c_vec = np.asarray(record["c_vec"], dtype=np.float64)
        sigma = float(record["sigma"])
        b_norm = float(record["b_norm"])
        beta_re = float(record["beta_re"])
        tau_re = float(record["tau_re"])
        zeta_vec = np.asarray(record["zeta_vec"], dtype=np.float64)
        delta_re_vec = np.asarray(record["delta_re_vec"], dtype=np.float64)
        omega_mat = np.asarray(record["omega_mat"], dtype=np.float64)

        t = c_vec * lam_vec
        s_norm_sq = (sigma * sigma) * beta_re
        s_norm_sq += float(np.sum(t * t))
        s_norm_sq += 2.0 * sigma * float(np.sum(t * zeta_vec))
        s_norm_sq += float(np.sum((t[:, None] * t[None, :]) * omega_mat))

        overlap = sigma * tau_re + float(np.sum(t * delta_re_vec))
        return float(s_norm_sq + (b_norm * b_norm) - 2.0 * overlap * b_norm), t

    def evaluate_total_loss(self, params: Mapping[str, np.ndarray]) -> float:
        total = 0.0
        for record in self._evaluate_local_term_cache(params):
            local_loss, _ = self._loss_from_record(record)
            total += local_loss

        return float(total)

    def evaluate_total_loss_and_exact_sigma_lambda_grads(
        self, params: Mapping[str, np.ndarray]
    ) -> tuple[float, np.ndarray, np.ndarray]:
        total = 0.0
        sigma_grad = np.zeros_like(params["sigma"], dtype=np.float64)
        lambda_grad = np.zeros_like(params["lambda"], dtype=np.float64)

        for record in self._evaluate_local_term_cache(params):
            entry = record["entry"]
            local_loss, t = self._loss_from_record(record)
            total += local_loss

            sigma = float(record["sigma"])
            b_norm = float(record["b_norm"])
            beta_re = float(record["beta_re"])
            tau_re = float(record["tau_re"])
            zeta_vec = np.asarray(record["zeta_vec"], dtype=np.float64)
            delta_re_vec = np.asarray(record["delta_re_vec"], dtype=np.float64)
            omega_mat = np.asarray(record["omega_mat"], dtype=np.float64)
            c_vec = np.asarray(record["c_vec"], dtype=np.float64)

            sigma_grad[entry.sys_id, entry.agent_id] += (
                2.0 * sigma * beta_re + 2.0 * float(np.dot(t, zeta_vec)) - 2.0 * b_norm * tau_re
            )

            dt = 2.0 * t + 2.0 * sigma * zeta_vec + 2.0 * (omega_mat @ t) - 2.0 * b_norm * delta_re_vec
            dlam_vec = c_vec * dt
            lambda_grad[entry.sys_id, entry.agent_id] += float(dlam_vec[0])
            for offset, nbr in enumerate(entry.neighbors, start=1):
                lambda_grad[entry.sys_id, nbr] += float(dlam_vec[offset])

        return float(total), sigma_grad.astype(np.float32), lambda_grad.astype(np.float32)

    def _eval_direct_operator(self, theta: np.ndarray, words: Sequence[PauliWord], coeffs: Sequence[float], name: str) -> float:
        tmpl = self._get_direct_operator_template(name, words, coeffs)
        pub = (tmpl.circuit, tmpl.observable_arg, [tmpl.pack(theta=theta)])
        res = self.estimator.run([pub]).result()[0]
        return float(np.sum(np.asarray(res.data.evs, dtype=np.float64)))

    def _eval_cross_operator(self, left: np.ndarray, right: np.ndarray, words: Sequence[PauliWord], coeffs: Sequence[float]) -> float:
        pubs = []
        coeff_list = []
        for word, coeff in zip(words, coeffs):
            tmpl = self._get_word_overlap_template(word if word.ops else None)
            pubs.append((tmpl.circuit, tmpl.observable_arg, [tmpl.pack(left=left, right=right)]))
            coeff_list.append(float(coeff))
        res = self.estimator.run(pubs).result()
        vals = [float(np.asarray(item.data.evs, dtype=np.float64).reshape(-1)[0]) for item in res]
        return float(np.dot(np.asarray(coeff_list, dtype=np.float64), np.asarray(vals, dtype=np.float64)))

    def _eval_b_overlap(self, b_spec: BStateSpec, theta: np.ndarray, words: Sequence[PauliWord], coeffs: Sequence[float]) -> float:
        pubs = []
        coeff_list = []
        for word, coeff in zip(words, coeffs):
            tmpl = self._get_bprep_overlap_template(b_spec, word if word.ops else None)
            pubs.append((tmpl.circuit, tmpl.observable_arg, [tmpl.pack(theta=theta)]))
            coeff_list.append(float(coeff))
        res = self.estimator.run(pubs).result()
        vals = [float(np.asarray(item.data.evs, dtype=np.float64).reshape(-1)[0]) for item in res]
        return float(np.dot(np.asarray(coeff_list, dtype=np.float64), np.asarray(vals, dtype=np.float64)))

    def _column_state_self(self, params: Mapping[str, np.ndarray], col: int, words: Sequence[PauliWord], coeffs: Sequence[float]) -> float:
        theta0 = params["alpha"][0, col]
        theta1 = params["alpha"][1, col]
        sigma0 = float(params["sigma"][0, col])
        sigma1 = float(params["sigma"][1, col])
        e00 = self._eval_direct_operator(theta0, words, coeffs, name=f"self_{col}")
        e11 = self._eval_direct_operator(theta1, words, coeffs, name=f"self_{col}")
        e01 = self._eval_cross_operator(theta0, theta1, words, coeffs)
        return 0.25 * (sigma0 * sigma0 * e00 + sigma1 * sigma1 * e11 + 2.0 * sigma0 * sigma1 * e01)

    def _column_cross(self, params: Mapping[str, np.ndarray], left_col: int, right_col: int) -> float:
        total = 0.0
        for r in range(2):
            for s in range(2):
                total += (
                    float(params["sigma"][r, left_col])
                    * float(params["sigma"][s, right_col])
                    * self._eval_cross_operator(
                        params["alpha"][r, left_col],
                        params["alpha"][s, right_col],
                        self._res_cross_words,
                        self._res_cross_coeffs,
                    )
                )
        return 0.25 * total

    def _b_to_column_overlap(
        self,
        params: Mapping[str, np.ndarray],
        row: int,
        col: int,
        words: Sequence[PauliWord],
        coeffs: Sequence[float],
    ) -> float:
        b_spec = self.system.b_specs[row][0]
        total = 0.0
        for r in range(2):
            total += float(params["sigma"][r, col]) * self._eval_b_overlap(
                b_spec,
                params["alpha"][r, col],
                words,
                coeffs,
            )
        return float(total / (2.0 * np.sqrt(2.0)))

    def compute_diagnostics(self, params: Mapping[str, np.ndarray]) -> Dict[str, float]:
        var_values = []
        for col in range(2):
            sigma0 = float(params["sigma"][0, col])
            sigma1 = float(params["sigma"][1, col])
            overlap = self._eval_cross_operator(
                params["alpha"][0, col],
                params["alpha"][1, col],
                [PauliWord((), "I")],
                [1.0],
            )
            var_values.append(0.25 * (sigma0 * sigma0 + sigma1 * sigma1 - 2.0 * sigma0 * sigma1 * overlap))

        self0 = self._column_state_self(params, 0, self._res_self_words, self._res_self_coeffs)
        self1 = self._column_state_self(params, 1, self._res_self_words, self._res_self_coeffs)
        cross = self._column_cross(params, 0, 1)
        b0d = self._b_to_column_overlap(params, 0, 0, self._diag_words, self._diag_coeffs)
        b0o = self._b_to_column_overlap(params, 0, 1, self._off_words, self._off_coeffs)
        b1o = self._b_to_column_overlap(params, 1, 0, self._off_words, self._off_coeffs)
        b1d = self._b_to_column_overlap(params, 1, 1, self._diag_words, self._diag_coeffs)

        residual_sq = self0 + self1 + 4.0 * cross + 1.0 - 2.0 * (b0d + b0o + b1o + b1d)
        residual_sq = max(0.0, float(residual_sq))
        return {
            "residual_norm": float(np.sqrt(residual_sq)),
            "variance_col0": float(var_values[0]),
            "variance_col1": float(var_values[1]),
            "variance_mean": float(np.mean(var_values)),
        }
