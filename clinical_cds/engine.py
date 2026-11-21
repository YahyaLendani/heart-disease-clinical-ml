"""Production Clinical Inference Engine v3: PLR + UniMP Masked Label Propagation + 3-Paradigm Ensemble."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .bundle import DEFAULT_BUNDLE_PATH, ClinicalModelBundle, compute_aha_bp_stage
from .schemas import (
    ClinicalPredictionResponse,
    DualClinicalExplanation,
    MatchedCohortNeighbor,
    OODReport,
    PatientClinicalInput,
    RecourseRecommendation,
)


def _hash_pseudo_id(pseudo_id: str | None) -> str | None:
    if not pseudo_id:
        return None
    return hashlib.sha256(pseudo_id.encode("utf-8")).hexdigest()[:16]


def compute_psi(expected_props: np.ndarray, actual_values: np.ndarray, bin_edges: np.ndarray) -> float:
    """Compute Population Stability Index (PSI) for live clinical drift monitoring."""
    if len(actual_values) == 0:
        return 0.0
    actual_counts, _ = np.histogram(actual_values, bins=bin_edges)
    actual_props = np.clip(actual_counts / max(len(actual_values), 1), 1e-6, 1.0)
    actual_props = actual_props / actual_props.sum()
    exp_props = np.clip(expected_props, 1e-6, 1.0)
    exp_props = exp_props / exp_props.sum()
    return float(np.sum((actual_props - exp_props) * np.log(actual_props / exp_props)))


class ClinicalInferenceEngine:
    """Thread-safe, fail-safe Stage-1 Primary Care Clinical Decision Support engine."""

    def __init__(
        self,
        bundle: ClinicalModelBundle,
        audit_log_path: str | Path | None = None,
    ) -> None:
        self.bundle = bundle
        self.device = next(bundle.model.parameters()).device
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        if self.audit_log_path is not None:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
        device: str = "auto",
        expected_sha256: str | None = None,
        audit_log_path: str | Path | None = None,
    ) -> ClinicalInferenceEngine:
        bundle = ClinicalModelBundle.load(
            bundle_path=bundle_path,
            device=device,
            expected_sha256=expected_sha256,
        )
        return cls(bundle=bundle, audit_log_path=audit_log_path)

    def _featurize_batch(
        self, patients: list[PatientClinicalInput]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert validated PatientClinicalInput objects to scaled v3 interaction features."""
        sbps = np.asarray([p.systolic_bp for p in patients], dtype=np.float32)
        dbps = np.asarray([p.diastolic_bp for p in patients], dtype=np.float32)
        aha_stages = compute_aha_bp_stage(sbps, dbps)

        cat_rows = []
        cont_rows = []
        has_v3_interactions = len(self.bundle.cat_cols) == 8

        for p, stage in zip(patients, aha_stages, strict=True):
            c_tier = p.cholesterol_level - 1
            g_tier = p.glucose_level - 1
            base_cat = [p.sex, c_tier, g_tier, p.smoke, p.alcohol, p.active]
            if has_v3_interactions:
                base_cat.extend([int(stage), min(4, c_tier + g_tier)])
            cat_rows.append(base_cat)

            base_cont = [
                p.age_years,
                p.height_cm,
                p.weight_kg,
                p.bmi,
                p.systolic_bp,
                p.diastolic_bp,
                p.pulse_pressure,
                p.mean_arterial_pressure,
            ]
            if has_v3_interactions:
                base_cont.extend([
                    (p.age_years * p.systolic_bp) / 100.0,
                    (p.bmi * p.age_years) / 100.0,
                    p.systolic_bp / max(p.diastolic_bp, 30.0),
                ])
            cont_rows.append(base_cont)

        x_cat = np.asarray(cat_rows, dtype=np.int64)
        x_cont_raw = np.asarray(cont_rows, dtype=np.float32)
        x_cont_scaled = (x_cont_raw - self.bundle.cont_mean) / np.where(
            self.bundle.cont_scale == 0, 1.0, self.bundle.cont_scale
        )
        x_tab = np.hstack([x_cat.astype(np.float32), x_cont_scaled]).astype(np.float32)
        return x_cat, x_cont_scaled.astype(np.float32), x_tab

    @torch.inference_mode()
    def _forward_patients(
        self, patients: list[PatientClinicalInput], return_explanations: bool = True
    ) -> dict:
        x_cat_np, x_cont_np, x_tab_np = self._featurize_batch(patients)

        x_cat_t = torch.from_numpy(x_cat_np).to(self.device)
        x_cont_t = torch.from_numpy(x_cont_np).to(self.device)

        self.bundle.model.eval()
        h_query, feat_attn = self.bundle.model.encode_tabular(x_cat=x_cat_t, x_cont=x_cont_t)

        # Learned-space exact cosine k-NN against stratified clinical medoid prototypes
        q_norm = F.normalize(h_query, p=2, dim=1, eps=1e-8)
        sims = torch.mm(q_norm, self.bundle.H_bank_norm.t())
        k_eff = min(self.bundle.k_neighbors, sims.size(1))
        topk_sims, topk_idx = torch.topk(sims, k=k_eff, dim=1)
        nbr_h_tab = self.bundle.H_tab_bank[topk_idx]
        nbr_labels = self.bundle.bank_outcomes_t[topk_idx]

        # Level-2 Inductive Ego-Graph GAT with UniMP Masked Label Propagation
        out = self.bundle.model.forward_minibatch(
            x_cat=x_cat_t,
            x_cont=x_cont_t,
            nbr_h_tab=nbr_h_tab,
            nbr_labels=nbr_labels,
            return_explanations=return_explanations,
        )
        if return_explanations:
            logits, expl = out
        else:
            logits, expl = out, None

        cal_logits = logits / max(self.bundle.temperature, 1e-3)
        gat_probs = torch.sigmoid(cal_logits).cpu().numpy().ravel()

        w_gat, w_xgb, w_hgb = self.bundle.ensemble_weights
        xgb_probs = (
            self.bundle.xgb_model.predict_proba(x_tab_np)[:, 1].astype(np.float64)
            if self.bundle.xgb_model is not None
            else None
        )
        hgb_probs = (
            self.bundle.hgb_model.predict_proba(x_tab_np)[:, 1].astype(np.float64)
            if self.bundle.hgb_model is not None
            else None
        )

        probs = w_gat * gat_probs
        total_w = w_gat
        if xgb_probs is not None:
            probs = probs + w_xgb * xgb_probs
            total_w += w_xgb
        if hgb_probs is not None:
            probs = probs + w_hgb * hgb_probs
            total_w += w_hgb
        probs = probs / max(total_w, 1e-8)
        if self.bundle.platt_a != 1.0 or self.bundle.platt_b != 0.0:
            clipped = np.clip(probs, 1e-5, 1.0 - 1e-5)
            z = self.bundle.platt_a * np.log(clipped / (1.0 - clipped)) + self.bundle.platt_b
            probs = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

        return {
            "probs": probs,
            "gat_probs": gat_probs,
            "xgb_probs": xgb_probs,
            "hgb_probs": hgb_probs,
            "x_cont_scaled": x_cont_np,
            "topk_sims": topk_sims.cpu().numpy(),
            "topk_idx": topk_idx.cpu().numpy(),
            "expl": expl,
        }

    def _build_mutable_recourse(
        self, patient: PatientClinicalInput, base_prob: float
    ) -> list[RecourseRecommendation]:
        candidates: list[tuple[str, dict[str, float | int], PatientClinicalInput]] = []

        if patient.systolic_bp > 125.0 or patient.diastolic_bp > 82.0:
            new_sbp = round(max(120.0, patient.systolic_bp - 20.0), 1)
            new_dbp = round(min(new_sbp - 15.0, max(78.0, patient.diastolic_bp - 10.0)), 1)
            changes = {"systolic_bp": new_sbp, "diastolic_bp": new_dbp}
            candidates.append((
                f"Antihypertensive BP control ({patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f} -> {new_sbp:.0f}/{new_dbp:.0f} mmHg)",
                changes,
                patient.model_copy(update=changes),
            ))

        if patient.cholesterol_level > 1:
            changes = {"cholesterol_level": 1}
            candidates.append((
                "Normalize total cholesterol tier (lipid-lowering therapy & diet)",
                changes,
                patient.model_copy(update=changes),
            ))

        if patient.glucose_level > 1:
            changes = {"glucose_level": 1}
            candidates.append((
                "Normalize fasting glucose tier (glycemic management)",
                changes,
                patient.model_copy(update=changes),
            ))

        if patient.bmi >= 25.0:
            new_w = round(max(45.0, patient.weight_kg * 0.93), 1)
            if new_w < patient.weight_kg - 1.0:
                changes = {"weight_kg": new_w}
                candidates.append((
                    f"7% clinical weight reduction ({patient.weight_kg:.1f} -> {new_w:.1f} kg)",
                    changes,
                    patient.model_copy(update=changes),
                ))

        if patient.active == 0:
            changes = {"active": 1}
            candidates.append((
                "Initiate regular aerobic physical activity (150 min/week)",
                changes,
                patient.model_copy(update=changes),
            ))

        if patient.smoke == 1:
            changes = {"smoke": 0}
            candidates.append((
                "Tobacco smoking cessation",
                changes,
                patient.model_copy(update=changes),
            ))

        combined_changes: dict[str, float | int] = {}
        for _, ch, _ in candidates[:3]:
            combined_changes.update(ch)
        if len(combined_changes) >= 2:
            candidates.append((
                "Combined cardiometabolic risk-factor optimization",
                combined_changes,
                patient.model_copy(update=combined_changes),
            ))

        if not candidates:
            return []

        cf_out = self._forward_patients([c[2] for c in candidates], return_explanations=False)
        recs: list[RecourseRecommendation] = []
        for (label, changes, _), p_new in zip(candidates, cf_out["probs"], strict=True):
            delta = float(base_prob - float(p_new))
            if delta > 0.005:
                recs.append(
                    RecourseRecommendation(
                        intervention=label,
                        changes=changes,
                        projected_risk_probability=round(float(p_new), 4),
                        absolute_risk_reduction=round(delta, 4),
                    )
                )

        recs.sort(key=lambda r: r.absolute_risk_reduction, reverse=True)
        return recs[:4]

    def _append_audit_record(
        self, patient: PatientClinicalInput, response: ClinicalPredictionResponse
    ) -> None:
        if self.audit_log_path is None:
            return
        record = {
            "stage": "stage1",
            "request_id": response.request_id,
            "timestamp_utc": response.timestamp_utc,
            "patient_id_hash": _hash_pseudo_id(patient.patient_pseudo_id),
            "model_version": response.model_version,
            "bundle_sha256": response.bundle_sha256[:16],
            "calibrated_risk_probability": response.calibrated_risk_probability,
            "model_disagreement": response.model_disagreement,
            "triage_decision": response.triage_decision,
            "conformal_alpha": response.conformal_alpha,
            "conformal_set": response.conformal_prediction_set,
            "is_ood": response.ood_report.is_ood,
            "max_abs_zscore": response.ood_report.max_abs_zscore,
            "min_neighbor_cosine_distance": response.ood_report.min_neighbor_cosine_distance,
            "latency_ms": response.latency_ms,
        }
        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _resolve_quantiles(self, alpha: float, sex: int) -> tuple[float, float, float]:
        akey = f"{alpha:.2f}"
        if akey in self.bundle.conformal_quantiles:
            qmap = self.bundle.conformal_quantiles[akey]
            used_alpha = alpha
        else:
            fallback_key = f"{self.bundle.default_conformal_alpha:.2f}"
            qmap = self.bundle.conformal_quantiles.get(
                fallback_key, next(iter(self.bundle.conformal_quantiles.values()))
            )
            used_alpha = self.bundle.default_conformal_alpha
        q0 = qmap.get(f"sex_{sex}_class_0", qmap["class_0"])
        q1 = qmap.get(f"sex_{sex}_class_1", qmap["class_1"])
        return used_alpha, q0, q1

    def predict_batch(
        self, patients: list[PatientClinicalInput]
    ) -> list[ClinicalPredictionResponse]:
        if not patients:
            return []

        t0 = time.perf_counter()
        fwd = self._forward_patients(patients, return_explanations=True)
        per_patient_ms = ((time.perf_counter() - t0) * 1000.0) / len(patients)

        feat_attns = fwd["expl"]["feature_attn"].cpu().numpy()
        nbr_attns = fwd["expl"]["neighbor_attn"].cpu().numpy()
        gates = fwd["expl"]["gate"].mean(dim=-1).cpu().numpy()

        responses: list[ClinicalPredictionResponse] = []
        now_utc = datetime.now(timezone.utc).isoformat()

        for i, patient in enumerate(patients):
            prob = float(fwd["probs"][i])
            gat_p = float(fwd["gat_probs"][i])
            xgb_p = float(fwd["xgb_probs"][i]) if fwd["xgb_probs"] is not None else None
            hgb_p = float(fwd["hgb_probs"][i]) if fwd["hgb_probs"] is not None else None

            active_preds = [p for p in (gat_p, xgb_p, hgb_p) if p is not None]
            disagreement = float(max(active_preds) - min(active_preds)) if len(active_preds) > 1 else 0.0

            max_z = float(np.max(np.abs(fwd["x_cont_scaled"][i])))
            min_cos_dist = float(1.0 - fwd["topk_sims"][i, 0])

            ood_reasons: list[str] = []
            if max_z > self.bundle.ood_zscore_threshold:
                ood_reasons.append(
                    f"Extreme standardized biomarker z-score ({max_z:.2f} > {self.bundle.ood_zscore_threshold:.1f})."
                )
            if min_cos_dist > self.bundle.ood_distance_threshold:
                ood_reasons.append(
                    f"Nearest learned-manifold clinical prototype distance ({min_cos_dist:.4f}) exceeds "
                    f"validation OOD threshold ({self.bundle.ood_distance_threshold:.4f})."
                )
            is_ood = len(ood_reasons) > 0
            ood_report = OODReport(
                is_ood=is_ood,
                max_abs_zscore=round(max_z, 3),
                min_neighbor_cosine_distance=round(min_cos_dist, 4),
                ood_distance_threshold=round(self.bundle.ood_distance_threshold, 4),
                reasons=ood_reasons,
            )

            used_alpha, q0, q1 = self._resolve_quantiles(patient.conformal_alpha, patient.sex)
            low_thresh = float(np.clip(1.0 - q1, 0.0, 1.0))
            high_thresh = float(np.clip(q0, 0.0, 1.0))
            cset: list[int] = []
            if prob <= q0:
                cset.append(0)
            if (1.0 - prob) <= q1:
                cset.append(1)
            if not cset:
                cset = [0, 1]

            nom_pct = int(round((1.0 - used_alpha) * 100))
            triage: Literal["LOW_RISK", "HIGH_RISK", "DEFER_TO_CLINICIAN"]
            if is_ood:
                triage = "DEFER_TO_CLINICIAN"
                triage_reason = "Automatic fail-safe abstention triggered by Out-of-Distribution alert."
            elif cset == [0]:
                triage = "LOW_RISK"
                triage_reason = (
                    f"Conformal set {{0}} at {nom_pct}% nominal coverage "
                    "(calibrated risk below sex-stratified low-risk boundary)."
                )
            elif cset == [1]:
                triage = "HIGH_RISK"
                triage_reason = (
                    f"Conformal set {{1}} at {nom_pct}% nominal coverage "
                    "(elevated cardiovascular risk — escalate to Stage-2 Cardiology Workup)."
                )
            else:
                triage = "DEFER_TO_CLINICIAN"
                triage_reason = (
                    f"Ambiguous conformal prediction set {{0, 1}} at {nom_pct}% nominal coverage — "
                    "defer to clinician review or escalate to Stage-2 Cardiology Workup."
                )

            biomarker_attentions = dict(
                sorted(
                    (
                        (name, round(float(w), 4))
                        for name, w in zip(self.bundle.feature_names, feat_attns[i], strict=True)
                    ),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            )
            neighbors: list[MatchedCohortNeighbor] = []
            for rank, (b_idx, cos_sim, g_w) in enumerate(
                zip(fwd["topk_idx"][i], fwd["topk_sims"][i], nbr_attns[i, 1:], strict=True)
            ):
                if rank >= 5:
                    break
                b_i = int(b_idx)
                age_n, sbp_n, bmi_n = self.bundle.bank_summary[b_i]
                neighbors.append(
                    MatchedCohortNeighbor(
                        cohort_index=b_i,
                        attention_weight=round(float(g_w), 4),
                        cosine_similarity=round(float(cos_sim), 4),
                        historical_outcome="CVD+" if int(self.bundle.bank_outcomes[b_i]) == 1 else "Healthy",
                        age_years=round(float(age_n), 1),
                        systolic_bp=round(float(sbp_n), 1),
                        bmi=round(float(bmi_n), 1),
                    )
                )

            dual_expl = DualClinicalExplanation(
                graph_reliance_gate=round(float(gates[i]), 4),
                biomarker_attentions=biomarker_attentions,
                top_cohort_neighbors=neighbors,
            )

            recourse = (
                self._build_mutable_recourse(patient, prob)
                if triage in ("HIGH_RISK", "DEFER_TO_CLINICIAN")
                else []
            )

            resp = ClinicalPredictionResponse(
                request_id=f"cds-{uuid.uuid4().hex[:12]}",
                timestamp_utc=now_utc,
                model_version=self.bundle.model_version,
                bundle_sha256=self.bundle.bundle_sha256,
                calibrated_risk_probability=round(prob, 4),
                gat_transformer_probability=round(gat_p, 4),
                xgboost_probability=round(xgb_p, 4) if xgb_p is not None else None,
                hgb_probability=round(hgb_p, 4) if hgb_p is not None else None,
                model_disagreement=round(disagreement, 4),
                triage_decision=triage,
                triage_reason=triage_reason,
                conformal_prediction_set=cset,
                conformal_alpha=used_alpha,
                conformal_low_threshold=round(low_thresh, 4),
                conformal_high_threshold=round(high_thresh, 4),
                ood_report=ood_report,
                dual_explanation=dual_expl,
                actionable_recourse=recourse,
                latency_ms=round(per_patient_ms, 2),
            )
            self._append_audit_record(patient, resp)
            responses.append(resp)

        return responses

    def predict(self, patient: PatientClinicalInput) -> ClinicalPredictionResponse:
        return self.predict_batch([patient])[0]
