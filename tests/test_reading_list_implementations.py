"""Unit tests for newly implemented architectures and methods from AI Thesis Reading List.

Tests:
1. TabTransformer (Huang et al., 2020 & Vaswani et al., 2017)
2. Graph Neural Networks: GCN & GAT (Kipf & Welling, 2017; Veličković et al., 2018)
3. Multimodal Classifier (Baltrušaitis et al., 2018; He et al., 2016; Dosovitskiy et al., 2020)
4. Temperature Scaling (Guo et al., 2017)
5. Deep Ensembles (Lakshminarayanan et al., 2017)
6. Causal Inference (Pearl & Mackenzie, 2018)
7. Grad-CAM (Selvaraju et al., 2017)
8. Fairness Mitigation (Barocas et al., 2020)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import (
    GradCAM,
    compare_observational_vs_causal,
    compute_propensity_scores,
    find_equal_opportunity_thresholds,
)
from models import (
    GATClassifier,
    GCNClassifier,
    ModelWithTemperature,
    MultimodalFusionClassifier,
    ResNet1DBlock,
    TabTransformer,
    ViTPatchEncoder,
    build_knn_patient_graph,
    compute_ece,
    normalize_adjacency_gcn,
)
from training import DeepEnsemble


class TestTabTransformer:
    def test_forward_output_shape_and_attention(self):
        batch_size = 8
        cat_dims = [3, 4, 2]
        cont_dim = 5
        model = TabTransformer(
            categorical_dims=cat_dims,
            continuous_dim=cont_dim,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            mlp_hidden_dims=[32],
        )

        x_cat = torch.randint(0, 2, (batch_size, len(cat_dims)))
        x_cont = torch.randn(batch_size, cont_dim)

        out, attn_weights = model(x_cat, x_cont, return_attn=True)
        assert out.shape == (batch_size, 1)
        assert len(attn_weights) == 1
        # Attention shape: (batch_size, num_heads, num_tokens, num_tokens)
        assert attn_weights[0].shape == (batch_size, 2, len(cat_dims), len(cat_dims))

    def test_gradient_flow(self):
        model = TabTransformer(
            categorical_dims=[2, 3],
            continuous_dim=4,
            embed_dim=8,
            num_heads=2,
            num_layers=1,
        )
        x_cat = torch.randint(0, 2, (4, 2))
        x_cont = torch.randn(4, 4)
        out = model(x_cat, x_cont)
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestGraphNeuralNetworks:
    def test_knn_graph_builder(self):
        features = np.random.randn(10, 4)
        adj = build_knn_patient_graph(features, k=3, metric="cosine")
        assert adj.shape == (10, 10)
        # Check symmetric
        assert torch.allclose(adj, adj.t())
        # Check self-loops
        assert torch.allclose(torch.diag(adj), torch.ones(10))

    def test_gcn_and_gat_forward(self):
        N, D = 12, 6
        x = torch.randn(N, D)
        adj = build_knn_patient_graph(x.numpy(), k=3)
        norm_adj = normalize_adjacency_gcn(adj)

        gcn = GCNClassifier(in_features=D, hidden_dim=16, out_features=1)
        out_gcn = gcn(x, norm_adj)
        assert out_gcn.shape == (N, 1)

        gat = GATClassifier(in_features=D, hidden_dim=8, num_heads=2, out_features=1)
        out_gat = gat(x, adj)
        assert out_gat.shape == (N, 1)


class TestMultimodalClassifier:
    def test_resnet_1d_block(self):
        block = ResNet1DBlock(in_channels=1, out_channels=8, stride=1)
        x = torch.randn(4, 1, 64)
        out = block(x)
        assert out.shape == (4, 8, 64)

    def test_vit_patch_encoder(self):
        vit = ViTPatchEncoder(signal_length=64, patch_size=16, embed_dim=32, num_heads=2, num_layers=1)
        x = torch.randn(4, 64)
        out = vit(x)
        assert out.shape == (4, 32)

    def test_multimodal_fusion_strategies(self):
        for strategy in ["early", "late", "cross_attention"]:
            model = MultimodalFusionClassifier(
                tabular_dim=10,
                signal_length=64,
                patch_size=16,
                embed_dim=32,
                fusion_strategy=strategy,
            )
            x_tab = torch.randn(4, 10)
            x_sig = torch.randn(4, 64)
            out = model(x_tab, x_sig)
            assert out.shape == (4, 1)


class TestCalibrationTemperatureScaling:
    def test_ece_computation(self):
        # Perfect calibration: probs == labels
        probs = np.array([0.1, 0.1, 0.9, 0.9])
        labels = np.array([0, 0, 1, 1])
        ece = compute_ece(probs, labels, n_bins=2)
        assert ece >= 0.0

    def test_temperature_scaling_optimization(self):
        base = nn.Linear(4, 1)
        model = ModelWithTemperature(base)
        val_logits = torch.randn(30, 1) * 3.0  # Overconfident
        val_labels = (torch.rand(30, 1) > 0.5).float()

        res = model.calibrate(val_logits, val_labels, max_iter=20)
        assert res["optimal_temperature"] > 0.0
        assert "ece_before" in res
        assert "ece_after" in res


class TestDeepEnsembles:
    def test_uncertainty_quantification(self):
        ensemble = DeepEnsemble(lambda: nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 1)), num_models=3)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        ensemble.fit(X, y, epochs=3, batch_size=10)
        res = ensemble.predict_with_uncertainty(X)

        assert res["mean_prob"].shape == (20,)
        assert res["epistemic_variance"].shape == (20,)
        assert (res["epistemic_variance"] >= 0.0).all()
        assert res["predictive_entropy"].shape == (20,)
        assert res["is_uncertain"].shape == (20,)


class TestCausalInference:
    def test_propensity_and_iptw(self):
        N = 50
        X = np.random.randn(N, 3)
        treatment = (X[:, 0] + np.random.randn(N) > 0).astype(int)
        outcome = (2.0 * treatment + X[:, 1] + np.random.randn(N) > 0).astype(int)

        props = compute_propensity_scores(X, treatment)
        assert len(props) == N
        assert (props > 0.0).all() and (props < 1.0).all()

        res = compare_observational_vs_causal(X, treatment, outcome, treatment_name="TestIntervention")
        assert "causal_ate_iptw" in res
        assert "observational_risk_difference" in res
        assert "confounding_bias" in res


class TestGradCAM:
    def test_gradcam_attribution(self):
        model = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
        )
        target_layer = model[0]  # First linear layer
        gradcam = GradCAM(model, target_layer)

        x = torch.randn(1, 6, requires_grad=True)
        cam = gradcam.generate_cam(x)
        assert len(cam) > 0
        assert (cam >= 0.0).all() and (cam <= 1.0).all()
        gradcam.close()


class TestFairnessMitigation:
    def test_equal_opportunity_tuning(self):
        N = 60
        y_true = np.random.randint(0, 2, N)
        y_prob = np.random.uniform(0, 1, N)
        sensitive = np.array([0] * 30 + [1] * 30)

        res = find_equal_opportunity_thresholds(y_true, y_prob, sensitive, target_tpr_tol=0.1)
        assert "mitigated_tpr_diff" in res
        assert "unmitigated_tpr_diff" in res
        assert "disparity_reduction_pct" in res
        assert res["mitigated_joint_f1"] >= 0.0
