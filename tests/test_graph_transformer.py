"""Unit and inductive zero-leakage tests for GraphAugmentedClinicalTransformer."""

from __future__ import annotations

import numpy as np
import torch

from models import (
    FeatureTokenizer,
    GraphAugmentedClinicalTransformer,
    InterpretableMultiHeadGATLayer,
    build_inductive_knn_indices,
    build_inductive_patient_graph,
)


class TestInductivePatientGraph:
    """Verify strict zero-leakage inductive graph properties."""

    def test_train_only_graph_symmetric_and_self_loops(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(16, 6)).astype(np.float32)
        adj = build_inductive_patient_graph(X_train, X_eval=None, k=4, metric="cosine")

        assert adj.shape == (16, 16)
        assert torch.allclose(adj, adj.t())
        assert torch.allclose(torch.diag(adj), torch.ones(16))

    def test_inductive_eval_graph_blocks_all_test_leakage(self):
        rng = np.random.default_rng(7)
        X_train = rng.normal(size=(20, 5)).astype(np.float32)
        X_eval = rng.normal(size=(8, 5)).astype(np.float32)
        k = 4

        adj = build_inductive_patient_graph(X_train, X_eval=X_eval, k=k, metric="cosine")
        assert adj.shape == (28, 28)

        # 1. Train-to-Eval block must be strictly 0 (eval never affects train embeddings)
        assert torch.all(adj[:20, 20:] == 0.0)

        # 2. Eval-to-Eval block must be strictly Identity (zero test-to-test leakage)
        assert torch.allclose(adj[20:, 20:], torch.eye(8))

        # 3. Every eval patient connects to exactly k training neighbors
        eval_to_train_degrees = adj[20:, :20].sum(dim=1)
        assert torch.allclose(eval_to_train_degrees, torch.full((8,), float(k)))


class TestGraphAugmentedClinicalTransformer:
    """Verify dual-level attention, gradient flow, inductive invariance, and explanations."""

    def test_feature_tokenizer_output_shape(self):
        tokenizer = FeatureTokenizer(categorical_dims=[2, 4], continuous_dim=5, embed_dim=16)
        x_cat = torch.randint(0, 2, (10, 2))
        x_cont = torch.randn(10, 5)
        tokens = tokenizer(x_cat=x_cat, x_cont=x_cont)
        # 1 [CLS] + 2 categorical + 5 continuous = 8 tokens
        assert tokens.shape == (10, 8, 16)

    def test_forward_shapes_and_dual_explanations(self):
        torch.manual_seed(42)
        n_nodes = 14
        x_cat = torch.stack([torch.randint(0, 2, (n_nodes,)), torch.randint(0, 4, (n_nodes,))], dim=1)
        x_cont = torch.randn(n_nodes, 6)
        adj = build_inductive_patient_graph(x_cont, k=3)

        model = GraphAugmentedClinicalTransformer(
            categorical_dims=[2, 4],
            continuous_dim=6,
            embed_dim=16,
            num_transformer_heads=4,
            num_transformer_layers=2,
            num_gat_heads=2,
            num_gat_layers=2,
        )
        logits, expl = model(x_cat, x_cont, adj, return_explanations=True)

        assert logits.shape == (n_nodes, 1)
        assert expl["feature_attn"].shape == (n_nodes, 8)
        assert expl["graph_attn"].shape == (n_nodes, n_nodes)
        assert expl["gate"].shape == (n_nodes, 16)

        # Feature attention rows sum to 1, gate strictly in (0, 1)
        assert torch.allclose(expl["feature_attn"].sum(dim=-1), torch.ones(n_nodes), atol=1e-5)
        assert torch.all((expl["gate"] > 0.0) & (expl["gate"] < 1.0))

    def test_gradient_flow_through_transformer_gat_and_gate(self):
        torch.manual_seed(0)
        x_cat = torch.randint(0, 2, (10, 2))
        x_cont = torch.randn(10, 4)
        adj = build_inductive_patient_graph(x_cont, k=3)

        model = GraphAugmentedClinicalTransformer(
            categorical_dims=[2, 2],
            continuous_dim=4,
            embed_dim=16,
            num_transformer_heads=2,
            num_transformer_layers=1,
            num_gat_heads=2,
            num_gat_layers=1,
        )
        loss = model(x_cat, x_cont, adj).sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"

    def test_inductive_invariance_eval_nodes_cannot_leak_into_train(self):
        """Perturbing test node features must produce bit-identical training logits."""
        torch.manual_seed(123)
        n_tr, n_ev = 12, 5
        x_cat_tr = torch.randint(0, 2, (n_tr, 2))
        x_cont_tr = torch.randn(n_tr, 4)
        x_cat_ev = torch.randint(0, 2, (n_ev, 2))
        x_cont_ev = torch.randn(n_ev, 4)

        adj_1 = build_inductive_patient_graph(x_cont_tr, x_cont_ev, k=3)
        x_cont_ev_perturbed = x_cont_ev * 10.0 + 5.0
        adj_2 = build_inductive_patient_graph(x_cont_tr, x_cont_ev_perturbed, k=3)

        model = GraphAugmentedClinicalTransformer(
            categorical_dims=[2, 2],
            continuous_dim=4,
            embed_dim=16,
            num_transformer_heads=2,
            num_transformer_layers=1,
            num_gat_heads=2,
            num_gat_layers=2,
        )
        model.eval()

        with torch.inference_mode():
            logits_1 = model(
                torch.cat([x_cat_tr, x_cat_ev], dim=0),
                torch.cat([x_cont_tr, x_cont_ev], dim=0),
                adj_1,
            )[:n_tr]
            logits_2 = model(
                torch.cat([x_cat_tr, x_cat_ev], dim=0),
                torch.cat([x_cont_tr, x_cont_ev_perturbed], dim=0),
                adj_2,
            )[:n_tr]

        assert torch.allclose(logits_1, logits_2, atol=1e-6)

    def test_explain_prediction_api(self):
        torch.manual_seed(99)
        x_cat = torch.randint(0, 2, (10, 2))
        x_cont = torch.randn(10, 3)
        adj = build_inductive_patient_graph(x_cont, k=4)
        feature_names = ["sex", "exang", "age", "thalach", "oldpeak"]

        model = GraphAugmentedClinicalTransformer(
            categorical_dims=[2, 2],
            continuous_dim=3,
            embed_dim=16,
            num_transformer_heads=2,
            num_transformer_layers=1,
            num_gat_heads=2,
            num_gat_layers=1,
        )
        expl = model.explain_prediction(
            x_cat=x_cat,
            x_cont=x_cont,
            adj=adj,
            patient_idx=0,
            feature_names=feature_names,
            top_k_neighbors=3,
        )

        assert 0.0 <= expl["risk_probability"] <= 1.0
        assert 0.0 <= expl["graph_reliance_gate"] <= 1.0
        assert set(expl["feature_importance"].keys()) == set(feature_names)
        assert len(expl["top_cohort_neighbors"]) == 3
        assert all(n["neighbor_idx"] != 0 for n in expl["top_cohort_neighbors"])


class TestScalableMiniBatchGraphTransformer:
    """Verify O(B * k) mini-batch inductive neighbor attention & exact GAT equivalence."""

    def test_inductive_knn_indices_zero_leakage(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(30, 6)).astype(np.float32)
        X_eval = rng.normal(size=(12, 6)).astype(np.float32)
        k = 5

        tr_idx, ev_idx = build_inductive_knn_indices(X_train, X_eval, k=k, metric="cosine", n_jobs=1)
        assert tr_idx.shape == (30, k)
        assert ev_idx is not None and ev_idx.shape == (12, k)

        # Training indices must never point to self
        for i in range(30):
            assert i not in tr_idx[i]

        # Eval indices must strictly point into training range [0, 30)
        assert int(ev_idx.min()) >= 0 and int(ev_idx.max()) < 30

    def test_forward_neighbors_matches_full_graph_gat_exactly(self):
        """1-hop ego-graph forward_neighbors must match full-graph GAT numerically."""
        torch.manual_seed(7)
        n_tr, n_ev, d, k = 16, 4, 16, 4
        h_tr = torch.randn(n_tr, d)
        h_ev = torch.randn(n_ev, d)

        # Build inductive adj and extract exact k neighbor indices for eval nodes
        adj = build_inductive_patient_graph(h_tr, h_ev, k=k, metric="cosine")
        gat = InterpretableMultiHeadGATLayer(embed_dim=d, num_heads=4, dropout=0.0)
        gat.eval()

        with torch.inference_mode():
            h_all = torch.cat([h_tr, h_ev], dim=0)
            full_out, full_attn = gat(h_all, adj)
            eval_full_out = full_out[n_tr:]

            # Gather exact training neighbors connected in adj[n_tr:, :n_tr]
            nbr_indices = torch.stack([torch.nonzero(adj[n_tr + i, :n_tr], as_tuple=False).squeeze(-1) for i in range(n_ev)])
            h_nbrs = h_tr[nbr_indices]  # (n_ev, k, d)
            mini_out, mini_attn = gat.forward_neighbors(h_ev, h_nbrs)

        assert torch.allclose(eval_full_out, mini_out, atol=1e-5)
        assert torch.allclose(mini_attn.sum(dim=-1), torch.ones(n_ev), atol=1e-5)

    def test_forward_minibatch_shapes_gradients_and_explanations(self):
        torch.manual_seed(11)
        B, k = 8, 5
        x_cat = torch.randint(0, 2, (B, 2))
        x_cont = torch.randn(B, 4)
        nbr_cat = torch.randint(0, 2, (B, k, 2))
        nbr_cont = torch.randn(B, k, 4)

        model = GraphAugmentedClinicalTransformer(
            categorical_dims=[2, 2],
            continuous_dim=4,
            embed_dim=16,
            num_transformer_heads=2,
            num_transformer_layers=1,
            num_gat_heads=2,
            num_gat_layers=2,
        )
        logits, expl = model.forward_minibatch(
            x_cat=x_cat,
            x_cont=x_cont,
            nbr_x_cat=nbr_cat,
            nbr_x_cont=nbr_cont,
            return_explanations=True,
            feature_noise_std=0.05,
        )
        assert logits.shape == (B, 1)
        assert expl["feature_attn"].shape == (B, 6)
        assert expl["neighbor_attn"].shape == (B, 1 + k)
        assert expl["gate"].shape == (B, 16)
        assert torch.allclose(expl["neighbor_attn"].sum(dim=-1), torch.ones(B), atol=1e-5)

        logits.sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing mini-batch gradient for {name}"

    def test_plr_and_unimp_label_propagation_gradients_and_isolation(self):
        """Verify PLR periodic embeddings & UniMP neighbor label propagation flow gradients with zero self-label leakage."""
        torch.manual_seed(21)
        B, k = 8, 5
        x_cat = torch.randint(0, 2, (B, 2))
        x_cont = torch.randn(B, 4)
        nbr_h_tab = torch.randn(B, k, 16)
        nbr_labels_0 = torch.zeros((B, k), dtype=torch.long)
        nbr_labels_1 = torch.ones((B, k), dtype=torch.long)

        # Standalone PLR tokenizer shape check
        tok = FeatureTokenizer(
            categorical_dims=[2, 2],
            continuous_dim=4,
            embed_dim=16,
            use_periodic_embeddings=True,
            n_frequencies=6,
        )
        assert tok(x_cat, x_cont).shape == (B, 7, 16)

        model = GraphAugmentedClinicalTransformer(
            categorical_dims=[2, 2],
            continuous_dim=4,
            embed_dim=16,
            num_transformer_heads=2,
            num_transformer_layers=1,
            num_gat_heads=2,
            num_gat_layers=2,
            use_periodic_embeddings=True,
            n_frequencies=6,
            use_label_propagation=True,
        )
        # Forward + backward with UniMP neighbor labels & label masking
        logits, expl = model.forward_minibatch(
            x_cat=x_cat,
            x_cont=x_cont,
            nbr_h_tab=nbr_h_tab,
            nbr_labels=nbr_labels_1,
            label_mask_prob=0.25,
            return_explanations=True,
        )
        assert logits.shape == (B, 1)
        assert expl["neighbor_attn"].shape == (B, 1 + k)
        logits.sum().backward()

        assert model.tokenizer.plr_freqs.grad is not None
        assert model.tokenizer.plr_weight.grad is not None
        assert model.label_embedding.weight.grad is not None

        # Neighbor outcome labels must influence logits (UniMP is active)
        model.eval()
        with torch.inference_mode():
            lg_0 = model.forward_minibatch(
                x_cat=x_cat, x_cont=x_cont, nbr_h_tab=nbr_h_tab, nbr_labels=nbr_labels_0
            )
            lg_1 = model.forward_minibatch(
                x_cat=x_cat, x_cont=x_cont, nbr_h_tab=nbr_h_tab, nbr_labels=nbr_labels_1
            )
            # When all neighbors are masked (id=2), output matches calling without nbr_labels
            lg_masked = model.forward_minibatch(
                x_cat=x_cat,
                x_cont=x_cont,
                nbr_h_tab=nbr_h_tab,
                nbr_labels=torch.full((B, k), 2, dtype=torch.long),
            )
            lg_none = model.forward_minibatch(
                x_cat=x_cat, x_cont=x_cont, nbr_h_tab=nbr_h_tab, nbr_labels=None
            )
        assert not torch.allclose(lg_0, lg_1)
        assert torch.allclose(lg_masked, lg_none, atol=1e-6)
