# TGC Baseline

This repository contains the current baseline implementation used before the
DCI-TGC reconstruction work. The baseline is based on Deep Temporal Graph
Clustering (TGC) and keeps the original two-stage clustering pipeline:

```text
node2vec pretraining
    -> TGC embedding optimization with temporal and structural losses
    -> KMeans evaluation on the learned node embeddings
```

The purpose of this repository is to provide a clean, version-controlled
starting point for analyzing, reproducing, and then extending the original TGC
baseline.

## Baseline Summary

The current method is an embedding-based temporal graph clustering baseline.
It does not directly output final cluster assignments from the model. Instead,
it learns/refines node embeddings and evaluates clustering quality by running
KMeans on those embeddings.

The baseline has three main stages:

1. Pretrain static node representations with node2vec.
2. Train TGC to refine node embeddings using temporal interaction prediction,
   DEC-style clustering regularization, and structural constraints.
3. Evaluate clustering quality with KMeans and report ACC, NMI, ARI, and F1.

In modeling terms, the baseline can be understood as:

```text
static topology pretraining
    + temporal interaction auxiliary objective
    + structural embedding regularization
    + post-hoc KMeans clustering
```

This is useful as a reference implementation, but it also exposes the main
motivation for later reconstruction: the training objective and the final
clustering objective are not fully aligned because final assignments still
depend on KMeans.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- docs/
|   |-- dynamic_prototype_tgc_technical_route.md
|   `-- dynamic_prototype_tgc_technical_route_zh.md
|-- framework/
|   |-- main.py
|   |-- model/
|   |   |-- DataSet.py
|   |   |-- TGCtrain.py
|   |   `-- evaluation.py
|   |-- pretrain/
|   |   |-- node2vec.py
|   |   |-- pretrain.py
|   |   `-- transform_data_to_edgelist.py
|   `-- experiments/
|       |-- clustering.py
|       `-- evaluation.py
|-- data/          # ignored by git
|-- emb/           # ignored by git
`-- all_results/   # ignored by git
```

Important files:

| File | Role |
|---|---|
| `framework/pretrain/pretrain.py` | Runs node2vec pretraining and writes `*_feature.emb` files. |
| `framework/model/DataSet.py` | Loads `(source, target, time)` events, builds historical neighbors, and samples negatives. |
| `framework/model/TGCtrain.py` | Main TGC training loop and loss computation. |
| `framework/model/evaluation.py` | KMeans-based clustering evaluation and diagnostics. |
| `framework/main.py` | Training entry point and command-line arguments. |
| `framework/experiments/clustering.py` | Standalone clustering evaluation on saved embeddings. |
| `docs/dynamic_prototype_tgc_technical_route.md` | Technical route report for the planned dynamic prototype reconstruction. |

## Data Format

Each dataset is expected under:

```text
data/<dataset>/
```

The baseline expects at least:

```text
data/<dataset>/<dataset>.txt
data/<dataset>/<dataset>.edgelist
data/<dataset>/node2label.txt
```

The temporal event file uses one event per line:

```text
source_node target_node timestamp
```

The label file uses:

```text
node_id label_id
```

The code assumes node ids are contiguous and start from `0`.

Datasets can be downloaded from the original Data4TGC source:

```text
https://github.com/MGitHubL/Data4TGC
```

Large datasets and generated artifacts are intentionally ignored by git:

```text
data/
emb/
all_results/
framework/pretrain/*_feature.emb
```

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The pinned dependencies are:

```text
torch==1.12.1
gensim==3.8.3
networkx==2.8.4
numpy==1.23.5
pandas==1.5.2
munkres==1.1.4
scikit-learn==1.1.1
```

The current training code uses CUDA tensors directly in several places. A CUDA
environment is therefore the expected runtime for the baseline.

## Pretraining

TGC uses node2vec embeddings as the initial node features for training.

Run pretraining from `framework/pretrain`:

```bash
cd framework/pretrain
python pretrain.py --data patent
```

The script reads:

```text
../../data/<dataset>/<dataset>.edgelist
```

and writes:

```text
./<dataset>_feature.emb
```

Default node2vec settings in `pretrain.py` include:

| Argument | Default |
|---|---:|
| `--dimensions` | 128 |
| `--walk-length` | 80 |
| `--num-walks` | 5 |
| `--window-size` | 10 |
| `--iter` | 1 |
| `--workers` | 8 |
| `--p` | 1 |
| `--q` | 1 |

These pretrained embeddings are part of the TGC baseline. They are not the
same as the raw feature files that may be distributed with the datasets.

## Training

Run training from `framework`:

```bash
cd framework
python main.py --dataset patent --clusters 6 --epoch 30
```

The default dataset in `main.py` is `patent`. Cluster counts are configured by
the built-in `k_dict`:

| Dataset | Clusters |
|---|---:|
| `patent` | 6 |
| `dblp` | 10 |
| `school` | 9 |
| `brain` | 10 |
| `arxivAI` | 5 |
| `arxivCS` | 40 |
| `arxivMath` | 31 |
| `arxivPhy` | 53 |
| `arxivLarge` | 172 |

Main training arguments:

| Argument | Default | Meaning |
|---|---:|---|
| `--dataset` | `patent` | Dataset name. |
| `--clusters` | dataset-specific | Number of clusters. |
| `--epoch` | 30 | Number of training epochs. |
| `--neg_size` | 5 | Negative samples per positive event. |
| `--hist_len` | 3 | Number of historical neighbors used by the temporal loss. |
| `--batch_size` | 128 | Mini-batch size. |
| `--learning_rate` | 0.01 | SGD learning rate. |
| `--emb_size` | 128 | Node embedding dimension. |
| `--directed` | `False` | Whether to treat input events as directed. |

The training loop also supports ablation flags:

| Flag | Effect |
|---|---|
| `--no_time_loss` | Disable temporal interaction loss. |
| `--no_node_loss` | Disable KL clustering loss. |
| `--no_res_st` | Disable source-target structural cosine loss. |
| `--no_res_sh` | Disable source-history structural cosine loss. |
| `--no_res_sn` | Disable source-negative structural cosine loss. |
| `--no_batch_loss` | Disable all three structural cosine losses. |

## Baseline Model Details

### 1. Node Embedding Table

The model keeps a trainable node embedding table:

```text
node_emb in R^{N x d}
```

It is initialized from the node2vec pretrained feature file:

```text
framework/pretrain/<dataset>_feature.emb
```

The original pretrained features are also kept as `pre_emb` and used to build
the target distribution for the clustering regularizer.

### 2. Cluster Layer

The baseline maintains a trainable cluster center matrix:

```text
cluster_layer in R^{K x d}
```

It is initialized by running KMeans on the pretrained node2vec embeddings.

During training, this layer participates in a DEC-style KL clustering loss.
However, final reported clustering results are still produced by running
KMeans on the learned `node_emb`, not by directly using `cluster_layer`.

### 3. Temporal Interaction Loss

For each event `(source, target, time)`, the dataset returns:

```text
source_node
target_node
target_time
history_nodes
history_times
history_masks
neg_nodes
```

The temporal score combines:

- source-target embedding distance,
- source-history attention,
- history-target compatibility,
- a learnable node-specific time factor `delta`,
- negative sampled nodes.

The positive event should receive a high score, while negative pairs should
receive low scores.

### 4. Structural Batch Loss

The structural part encourages:

- source and target nodes to be close,
- source and historical neighbors to be close,
- source and negative nodes to be far apart.

This is implemented with cosine-similarity constraints:

```text
res_st: source-target
res_sh: source-history
res_sn: source-negative
```

### 5. KL Clustering Loss

The clustering regularizer follows a DEC-style target distribution:

```text
q = Student-t similarity between node embedding and cluster centers
p = sharpened target distribution derived from pretrained embeddings
loss = KL(q || p)
```

This helps shape the embedding space, but it does not remove the dependency on
post-hoc KMeans for final evaluation.

### 6. Total Loss

The implemented objective can be summarized as:

```text
L = L_time + L_node + L_batch
```

where:

```text
L_time  = temporal positive/negative interaction loss
L_node  = DEC-style KL clustering loss
L_batch = structural cosine constraints
```

Each component can be disabled with the ablation flags listed above.

## Evaluation

During each epoch, `TGCtrain.py` evaluates the current node embeddings with
KMeans:

```text
node_emb -> KMeans(K) -> cluster_id -> metrics
```

If labels are available, the baseline reports:

```text
ACC
NMI
ARI
F1
```

The current evaluation code also records diagnostic values:

```text
silhouette
Davies-Bouldin index
Calinski-Harabasz score
empty cluster count
max cluster ratio
cluster size coefficient of variation
epoch-to-epoch NMI
cluster switch rate
center shift
```

Training logs are written to:

```text
all_results/<dataset>/<dataset>_TGC_<epoch>_<timestamp>_all.txt
```

Best embeddings are written to:

```text
emb/<dataset>/<dataset>_TGC_<epoch>.emb
```

Create the output directory before training if it does not exist:

```bash
mkdir -p ../emb/patent
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force ..\emb\patent
```

## Standalone Clustering

Saved embeddings can be evaluated separately with:

```bash
cd framework/experiments
python clustering.py
```

This script also uses KMeans and the same clustering metrics.

## Known Baseline Limitations

This repository intentionally preserves the baseline design before the planned
dynamic prototype reconstruction. The most important limitations are:

1. The model is still a two-stage clustering pipeline.

   Training optimizes embeddings and auxiliary losses, while final clustering
   depends on KMeans.

2. Time is used as an auxiliary signal for embedding learning.

   The baseline does not explicitly model clusters as time-evolving objects.

3. Random mini-batches break event-stream causality.

   The current DataLoader uses shuffled batches. This is acceptable for the
   original embedding optimization setup, but not for future memory-based
   temporal event models.

4. Connected nodes are generally encouraged to become close.

   This works better for homophilous datasets, but it is less suitable for
   datasets where cross-cluster interactions are important.

5. Negative sampling uses a large fixed table.

   `DataSet.py` builds a negative sampling table of size `1e8`, which costs
   substantial CPU memory.

6. Full KMeans evaluation is run repeatedly.

   This can become expensive on large datasets.

These limitations are the basis for the planned DCI-TGC route documented under
`docs/`.

## Planned Reconstruction Direction

The technical reports in `docs/` propose moving from the current baseline to an
end-to-end temporal graph clustering method:

```text
TGN-style node memory
    + dynamic cluster prototypes C(t)
    + dynamic cluster interaction matrix B(t)
    + chronological window training
    + direct cluster assignment q_i(t)
```

The recommended migration path is:

1. Replace post-hoc KMeans with an end-to-end static prototype assignment.
2. Introduce chronological or time-window training.
3. Add dynamic cluster prototypes `C(t)`.
4. Add dynamic cluster interaction matrix `B(t)`.
5. Evaluate with both clustering metrics and temporal/event diagnostics.

## Related Papers

This repository starts from the public TGC codebase and related benchmark work:

```bibtex
@inproceedings{TGC_ML_ICLR,
  author={Liu, Meng and Liu, Yue and Liang, Ke and Tu, Wenxuan and Wang, Siwei and Zhou, Sihang and Liu, Xinwang},
  title={Deep Temporal Graph Clustering},
  booktitle={The 12th International Conference on Learning Representations},
  year={2024}
}

@ARTICLE{BenchTGC_ML_TPAMI,
  author={Liu, Meng and Liang, Ke and Wang, Siwei and Hu, Xingchen and Zhou, Sihang and Liu, Xinwang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Deep Temporal Graph Clustering: A Comprehensive benchmark and Datasets},
  year={2025}
}

@ARTICLE{MVTGC_ML_TNNLS,
  author={Liu, Meng and Liang, Ke and Yu, Hao and Meng, Lingyuan and Wang, Siwei and Zhou, Sihang and Liu, Xinwang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Multiview Temporal Graph Clustering},
  year={2025},
  pages={1-14}
}
```
