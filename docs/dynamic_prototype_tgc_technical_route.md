# Dynamic Prototype Temporal Graph Clustering: Technical Route Report

## 1. Background and Baseline Understanding

The current project provides a baseline implementation of Deep Temporal Graph Clustering (TGC). Its core pipeline is a two-stage clustering framework:

1. Pretrain node embeddings with node2vec.
2. Use TGC training losses to refine embeddings.
3. Run KMeans during or after training to evaluate clustering quality.

From a modeling perspective, the current TGC framework can be summarized as:

```text
Static topology pretraining + temporal interaction auxiliary loss + post-hoc clustering
```

The current implementation mainly uses temporal information to improve node representations. The cluster structure itself is not explicitly modeled as a dynamic object. The final cluster assignment still depends heavily on KMeans, so the training objective and the final clustering objective are not fully aligned.

The proposed research direction is:

```text
End-to-end temporal graph clustering with dynamic cluster prototypes and dynamic cluster interactions
```

This route chooses:

- Route A as the implementation foundation: convert two-stage clustering into end-to-end clustering.
- Route B as the main innovation: model cluster structures as time-evolving objects.

## 2. Problems in the Current Baseline

### 2.1 Two-Stage Objective Mismatch

The baseline first obtains pretrained features from node2vec and then optimizes temporal and structural losses. However, final clustering still relies on KMeans. This creates a mismatch:

```text
Training objective: temporal interaction prediction + embedding regularization
Evaluation objective: cluster quality after KMeans
```

The model does not directly optimize cluster assignment during training.

### 2.2 Temporal Information Is Only an Auxiliary Signal

The baseline uses time to model interaction likelihood, but the cluster structure remains static. In temporal graphs, clusters may evolve:

- A research field may drift over years.
- A social group may show different interaction patterns at different time periods.
- Brain functional areas may interact across clusters instead of only within clusters.

Therefore, time should not only regularize node embeddings. It should also participate in the generation of cluster semantics and cluster-level interactions.

### 2.3 Random Batch Training Breaks Temporal Causality

The current code uses shuffled mini-batches. This is acceptable for static embedding optimization, but it is problematic for event-stream models such as memory networks or temporal point processes. For event data, the model should only use events before time `t` when predicting an event at time `t`.

The new method should use chronological mini-batches or time-window mini-batches.

### 2.4 Topology Homophily Is Not Universal

The baseline implicitly encourages connected nodes to be close. This works for homophilous datasets, but not all event interaction graphs are purely homophilous. Some datasets may contain strong cross-cluster interactions. A model that only pulls connected nodes into the same region may fail to explain such data.

## 3. Dataset Characteristics in This Project

The project datasets have very different structural and temporal properties.

| Dataset | Nodes | Events | Time Granularity | Repeated Directed Edges | Same-Label Edge Ratio | Main Characteristic |
|---|---:|---:|---:|---:|---:|---|
| patent | 12,214 | 41,916 | 891 | 0.0% | 67.0% | Sparse, almost no repeated pair interactions |
| dblp | 28,085 | 236,894 | 27 | 31.4% | 62.4% | Medium repeated interaction, strong homophily |
| arXivAI | 69,854 | 699,206 | 27 | 0.0% | 73.0% | Citation-like, strong structure, highly imbalanced time distribution |
| brain | 5,000 | 1,955,488 | 12 | 9.9% directed, 54.8% undirected | 25.3% | Dense, low label homophily, likely cross-cluster interactions |
| school | 327 | 188,508 | 7375 | 96.9% | 93.5% | Small-scale, fine-grained, high-frequency repeated contacts |

These statistics imply that one single temporal mechanism may not fit all datasets.

### 3.1 Sparse Citation-Like Graphs

Datasets such as `patent` and `arXivAI` have almost no repeated directed edges. Pair-level Hawkes processes are not ideal here because there are not enough repeated pair interactions to estimate self-excitation. For these datasets, time is better interpreted as:

- topic evolution,
- temporal stage,
- distribution shift,
- cluster semantic drift.

Dynamic cluster prototypes are especially important.

### 3.2 Repeated Interaction Graphs

Datasets such as `school` have fine-grained time and highly repeated interactions. These data fit temporal point process or Hawkes-like intensity modeling. Recent contacts can strongly influence future contacts.

### 3.3 Dense Cross-Cluster Interaction Graphs

The `brain` dataset has low same-label edge ratio. This suggests that edges may represent functional communication across different regions, not simply within-cluster similarity. For such data, a pure homophily-based clustering loss is insufficient. The model needs a cluster interaction matrix to represent cross-cluster event generation.

## 4. First-Principles Modeling of Event Interaction Graphs

The basic observation unit is:

```text
e = (u, v, t)
```

The model should explain why node `u` and node `v` interact at time `t`.

From first principles, event generation depends on at least three factors:

1. Long-term topology preference:
   which nodes or groups are structurally related.

2. Short-term temporal excitation:
   how recent events change future interaction probability.

3. Cluster-level interaction pattern:
   which latent groups tend to interact at a given time.

Thus, the proposed method should model both node-level states and cluster-level structures.

## 5. Proposed Core Idea

The proposed method introduces two dynamic cluster structures:

```text
C(t): time-evolving cluster prototypes
B(t): time-evolving cluster interaction matrix
```

Their roles are different:

```text
C(t): what each cluster currently means
B(t): how clusters currently interact
```

This distinction is important. `C(t)` models cluster semantics, while `B(t)` models cluster-level event generation.

## 6. Dynamic Cluster Prototype C(t)

Assume there are `K` clusters and the embedding dimension is `d`.

```text
C(t) in R^{K x d}
C_k(t) in R^d
```

`C_k(t)` is the prototype of cluster `k` at time `t`.

Unlike static clustering, temporal graph clustering should allow cluster semantics to evolve:

```text
C_k(t) = C_k^base + Delta C_k(t)
```

where:

- `C_k^base` captures the long-term stable semantics of cluster `k`.
- `Delta C_k(t)` captures temporal drift.

### 6.1 Prototype Memory Update

A practical implementation is to maintain a memory vector for each cluster:

```text
m_k(t) = GRU(m_k(t-1), r_k(t))
C_k(t) = W_c m_k(t)
```

The input `r_k(t)` is a soft aggregation of node representations assigned to cluster `k` in the current time window:

```text
r_k(t) = sum_i q_i,k(t) z_i(t) / sum_i q_i,k(t)
```

where:

- `z_i(t)` is the dynamic representation of node `i`.
- `q_i,k(t)` is the soft probability that node `i` belongs to cluster `k`.

### 6.2 End-to-End Cluster Assignment

The cluster assignment is computed directly from dynamic prototypes:

```text
q_i,k(t) = softmax(-||z_i(t) - C_k(t)||^2 / tau)
```

This replaces post-hoc KMeans with a trainable assignment mechanism.

## 7. Dynamic Cluster Interaction Matrix B(t)

Dynamic prototypes alone assume that events are mainly explained by node similarity or cluster membership. This is not enough for cross-cluster interaction datasets.

The model therefore introduces:

```text
B(t) in R^{K x K}
B_k,l(t): interaction tendency between cluster k and cluster l at time t
```

For two nodes `u` and `v`, their cluster-level interaction score is:

```text
s_cluster(u, v, t) = q_u(t)^T B(t) q_v(t)
```

Expanded form:

```text
s_cluster(u, v, t) = sum_k sum_l q_u,k(t) B_k,l(t) q_v,l(t)
```

This means that if `u` belongs mostly to cluster `k` and `v` belongs mostly to cluster `l`, their interaction probability depends on `B_k,l(t)`.

### 7.1 Event Intensity with C(t) and B(t)

The event probability or intensity can be modeled as:

```text
lambda_uv(t) = sigmoid(
    node_affinity(z_u(t), z_v(t))
    + q_u(t)^T B(t) q_v(t)
    + time_effect(delta_t)
    + node_bias(u, v)
)
```

where:

- `node_affinity` captures node-level compatibility.
- `q_u(t)^T B(t) q_v(t)` captures cluster-level interaction.
- `time_effect(delta_t)` captures temporal decay or excitation.
- `node_bias` captures node popularity.

### 7.2 Dynamic B(t) Update

`B(t)` can also have a stable part and a dynamic part:

```text
B(t) = B^base + Delta B(t)
```

The dynamic part can be generated from the observed soft cluster interaction matrix in the current window:

```text
A_cluster(t) = sum_(u,v,t in window) q_u(t) q_v(t)^T
```

Then:

```text
h_B(t) = GRU(h_B(t-1), A_cluster(t))
B(t) = W_B h_B(t)
```

This allows the model to learn changing cluster interaction patterns over time.

## 8. Temporal Module Choice

The temporal module should not be fixed blindly. The project datasets suggest different temporal mechanisms.

### 8.1 Hawkes-Like Temporal Point Process

Suitable for:

- fine-grained event streams,
- repeated pair interactions,
- strong short-term self-excitation.

Best fit:

- `school`

Less suitable for:

- `patent`,
- `arXivAI`,
- other citation-like graphs with nearly no repeated directed edges.

### 8.2 TGN-Style Node Memory

Suitable for:

- general event streams,
- both repeated and non-repeated interactions,
- node state evolution.

A TGN-style memory mechanism is a more general foundation than a pure Hawkes model. Each node keeps a memory state:

```text
s_i(t) = MemoryUpdate(s_i(t-1), message_i(t))
z_i(t) = Encoder(s_i(t), temporal_neighbors)
```

This memory can feed both `C(t)` and `B(t)`.

### 8.3 Recommended Choice

Use:

```text
TGN-style node memory + cluster-conditioned neural event intensity
```

This gives a unified framework:

- For repeated-contact datasets, the intensity module behaves like a neural point process.
- For sparse citation-like datasets, it behaves like temporal window link prediction.
- For dense cross-cluster datasets, `B(t)` explains cluster-level interactions.

## 9. Training and Batching Strategy

The new method should not use random shuffled batches for memory-based temporal modeling.

Recommended batching:

```text
sort events by time
split into time windows
process each window chronologically
use current memory to predict events
update node memory and cluster memory after prediction
```

This prevents temporal information leakage.

### 9.1 Window-Level Training Flow

For each time window:

1. Load positive events in the window.
2. Sample negative node pairs.
3. Compute node states `z_i(t)` from memory.
4. Compute dynamic prototypes `C(t)`.
5. Compute assignments `q_i(t)`.
6. Compute dynamic interaction matrix `B(t)`.
7. Predict event intensity for positive and negative pairs.
8. Compute losses.
9. Update parameters.
10. Update node memory, prototype memory, and interaction memory.

## 10. Loss Design

The overall loss can be:

```text
L = L_event
  + lambda_1 L_proto
  + lambda_2 L_cluster_interaction
  + lambda_3 L_temporal_smooth
  + lambda_4 L_balance
  + lambda_5 L_contrast
```

### 10.1 Event Prediction Loss

Positive events should have high intensity, negative sampled pairs should have low intensity:

```text
L_event = -log lambda_uv(t) - sum_neg log(1 - lambda_un(t))
```

### 10.2 Prototype Clustering Loss

Nodes should be close to their assigned dynamic prototypes:

```text
L_proto = sum_i sum_k q_i,k(t) ||z_i(t) - C_k(t)||^2
```

### 10.3 Cluster Interaction Loss

Observed events should be explainable by `B(t)`:

```text
L_cluster_interaction = BCE(q_u(t)^T B(t) q_v(t), y_uv)
```

### 10.4 Temporal Smoothness Loss

Cluster prototypes and interaction matrices should not drift arbitrarily:

```text
L_smooth_C = sum_k ||C_k(t) - C_k(t-1)||^2
L_smooth_B = ||B(t) - B(t-1)||_F^2
```

The smoothness weight can depend on time interval:

```text
w(delta_t) = exp(-delta_t / sigma)
```

Short intervals enforce stronger smoothness; long intervals allow more drift.

### 10.5 Balance Loss

To prevent all nodes from collapsing into one cluster:

```text
p_k = mean_i q_i,k(t)
L_balance = KL(p || uniform)
```

### 10.6 Contrastive Structural Loss

Keep useful topological discrimination:

```text
positive: observed temporal neighbors
negative: sampled non-interacting nodes
```

This preserves the strength of the original TGC structural objective while integrating it into end-to-end clustering.

## 11. Hardware and Scalability Considerations

The current baseline has several hidden costs:

- A fixed negative sampling table of size `1e8`, about 763 MB CPU memory.
- Full-batch KMeans evaluation every epoch.
- Full node embedding table resident on GPU.

Approximate current embedding-related memory:

| Dataset | Node Tensor Memory | KMeans Input Memory |
|---|---:|---:|
| patent | about 18 MB | about 12 MB |
| dblp | about 41 MB | about 27 MB |
| arXivAI | about 103 MB | about 68 MB |
| arXivPhy | about 1.23 GB | about 818 MB |
| arXivLarge | about 1.95 GB | about 1.29 GB |
| brain | about 7 MB | about 5 MB |
| school | less than 1 MB | less than 1 MB |

The proposed memory-based method adds at least one additional `N x d` node memory table. This is manageable for:

- `patent`,
- `dblp`,
- `school`,
- `brain`,
- `arXivAI`.

For very large datasets such as `arXivLarge` and `arXivPhy`, training may require:

- larger GPU memory,
- CPU/offloaded node memory,
- neighbor sampling,
- time-window sampling,
- less frequent full evaluation.

Recommended first-stage experimental datasets:

```text
patent + dblp + school + brain
```

These cover sparse citation-like graphs, medium homophilous graphs, fine-grained contact networks, and dense cross-cluster interaction graphs.

## 12. Experimental Plan

### 12.1 Baselines

Compare against:

- original TGC,
- node2vec + KMeans,
- static graph clustering methods if available,
- end-to-end static prototype variant,
- dynamic prototype without `B(t)`.

### 12.2 Ablation Studies

Required ablations:

1. Static prototype instead of `C(t)`.
2. Remove `B(t)`.
3. Static `B` instead of dynamic `B(t)`.
4. Remove temporal smoothness.
5. Remove balance loss.
6. Use post-hoc KMeans on learned embeddings.
7. Random batch vs chronological window batch.

### 12.3 Metrics

Standard clustering metrics:

- ACC,
- NMI,
- ARI,
- F1.

Temporal and structural diagnostics:

- assignment entropy,
- cluster balance,
- prototype drift,
- cluster interaction drift,
- node cluster switch rate,
- event prediction AUC/AP,
- homophily-aware and cross-cluster interaction analysis.

## 13. Main Research Claims

The method can be positioned around the following claims:

1. Temporal graph clustering should not treat time only as an auxiliary representation signal.
2. In event interaction graphs, clusters are dynamic objects whose semantics and interactions evolve over time.
3. `C(t)` captures evolving cluster semantics.
4. `B(t)` captures evolving cluster-level event generation.
5. End-to-end assignment aligns the training objective with the clustering objective and avoids post-hoc KMeans dependence.

## 14. Suggested Method Name

Possible names:

- Dynamic Prototype Temporal Graph Clustering (DPTGC)
- End-to-End Dynamic Prototype Temporal Clustering (E2DPTC)
- Dynamic Cluster Interaction Temporal Graph Clustering (DCI-TGC)

The strongest name for the current technical route is:

```text
Dynamic Cluster Interaction Temporal Graph Clustering (DCI-TGC)
```

because it highlights both `C(t)` and `B(t)`.

## 15. Summary

The proposed route uses the current TGC project as a baseline but does not remain constrained by its two-stage structure. The new technical direction is to build an end-to-end temporal graph clustering model where:

- node states evolve through temporal memory,
- cluster prototypes `C(t)` evolve over time,
- cluster interaction matrix `B(t)` explains cross-cluster event generation,
- training follows chronological event windows,
- clustering assignment is produced directly by the model.

This route provides a clearer first-principles explanation of event interaction data and gives stronger innovation than simply replacing KMeans with a clustering head.
