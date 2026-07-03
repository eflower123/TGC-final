# 动态原型时间图聚类技术路线报告

## 1. 研究背景

当前项目是 Deep Temporal Graph Clustering（TGC）的 baseline。其核心流程可以概括为：

```text
node2vec 预训练节点表示
    -> TGC 使用时间交互与结构损失微调 embedding
    -> 每轮或训练后使用 KMeans 评估聚类结果
```

因此，当前 TGC 本质上仍是两阶段聚类：

```text
静态拓扑预训练 + 时间辅助表示学习 + 后验 KMeans 聚类
```

这带来两个核心问题：

1. 训练目标与聚类目标不完全一致。模型训练时优化的是时间交互预测和结构约束，最终聚类却依赖 KMeans。
2. 时间只作为节点表示学习的辅助信号，簇结构本身没有被建模为随时间演化的对象。

本技术路线选择：

```text
A 路线作为实现基础：两阶段聚类 -> 端到端聚类训练
B 路线作为主创新：静态簇结构 -> 时间演化簇结构
```

建议方法名称：

```text
Dynamic Cluster Interaction Temporal Graph Clustering, DCI-TGC
```

核心思想是同时学习：

```text
C(t)：动态簇原型，表示“每个簇在时间 t 是什么”
B(t)：动态簇交互矩阵，表示“哪些簇在时间 t 更容易发生交互”
```

## 2. 当前 Baseline 的代码框架

当前项目的主要代码结构如下：

- `framework/pretrain/pretrain.py`：使用 node2vec 生成预训练 embedding。
- `framework/main.py`：训练入口。
- `framework/model/DataSet.py`：读取 `(source, target, time)` 事件，构造历史邻居和负采样。
- `framework/model/TGCtrain.py`：TGC 主训练逻辑，包括时间损失、KL 聚类损失和结构损失。
- `framework/model/evaluation.py`：每轮使用 KMeans 计算 ACC、NMI、ARI、F1 等指标。

当前 TGC 的损失大致包括：

```text
L = L_time + L_node + L_batch
```

其中：

- `L_time`：基于历史交互和时间差的时间交互预测损失。
- `L_node`：类似 DEC 的 KL 聚类自训练损失。
- `L_batch`：source-target、source-history、source-negative 的结构余弦约束。

但是，当前聚类结果仍然依赖 KMeans，因此不是真正的端到端聚类模型。

## 3. 数据集结构特点

当前项目中的数据集不是同一类动态图。它们在节点规模、时间粒度、重复交互、同质性上差异很大。

| 数据集 | 节点数 | 事件数 | 时间粒度 | 有向重复边比例 | 同类边比例 | 主要特点 |
|---|---:|---:|---:|---:|---:|---|
| patent | 12,214 | 41,916 | 891 | 0.0% | 67.0% | 稀疏，几乎无重复交互 |
| dblp | 28,085 | 236,894 | 27 | 31.4% | 62.4% | 中等重复，结构同质性较强 |
| arXivAI | 69,854 | 699,206 | 27 | 0.0% | 73.0% | 类 citation 图，结构强，时间分布不均衡 |
| brain | 5,000 | 1,955,488 | 12 | 9.9% 有向，54.8% 无向 | 25.3% | 稠密，低同质性，可能存在跨簇功能交互 |
| school | 327 | 188,508 | 7375 | 96.9% | 93.5% | 小规模，细粒度，高频重复接触 |

这些统计说明：不能简单地固定采用 Hawkes 过程、TGN memory 或原 TGC 批处理机制。不同数据集对应不同事件生成机制。

## 4. 第一性原理：如何理解事件交互数据

时间图中的基本观测单位是：

```text
e = (u, v, t)
```

问题不是简单地问“哪些节点相似”，而是问：

```text
为什么节点 u 和节点 v 会在时间 t 发生交互？
```

从第一性原理看，事件交互至少由三类因素共同决定：

1. **长期拓扑偏好**
   某些节点或群体在长期结构上更接近。

2. **短期时间激发**
   最近发生的事件会提高或降低未来交互概率。

3. **簇级交互结构**
   事件不一定只发生在同簇节点之间。某些数据中，跨簇交互本身就是主要信号。

因此，新模型不能只做“连接节点拉近、非连接节点推远”。它还需要显式建模：

```text
节点状态如何随时间变化
簇语义如何随时间变化
簇与簇之间的交互强度如何随时间变化
```

## 5. 总体技术路线

建议的新方法框架如下：

```text
事件流 (u, v, t)
    -> 按时间排序 / 时间窗口批处理
    -> TGN-style 节点记忆编码器
    -> 动态节点表示 z_i(t)
    -> 动态簇原型 C(t)
    -> 软簇分配 q_i(t)
    -> 动态簇交互矩阵 B(t)
    -> 事件强度预测 lambda_uv(t)
    -> 端到端联合优化
```

核心变化是：

```text
原 TGC：学习 embedding，再用 KMeans 聚类
新方法：模型内部直接学习 q_i(t)、C(t)、B(t)
```

## 6. 动态簇原型 C(t)

设有 `K` 个簇，embedding 维度为 `d`：

```text
C(t) ∈ R^{K × d}
C_k(t) ∈ R^d
```

`C_k(t)` 表示第 `k` 个簇在时间 `t` 的原型中心。

静态聚类中，簇中心是固定的。但在时间图中，簇语义可能演化。例如：

- DBLP 中一个研究领域会随年份从传统方法转向深度学习。
- School 中同一群学生在上课、午休、放学时有不同交互模式。
- Patent/arXiv 中主题簇随时间发生漂移。

因此可以定义：

```text
C_k(t) = C_k^base + ΔC_k(t)
```

其中：

- `C_k^base` 表示簇的长期稳定语义。
- `ΔC_k(t)` 表示时间带来的短期漂移。

### 6.1 C(t) 的记忆更新

可以为每个簇维护一个 memory：

```text
m_k(t) = GRU(m_k(t-1), r_k(t))
C_k(t) = W_c m_k(t)
```

其中 `r_k(t)` 是当前时间窗口中属于簇 `k` 的节点表示聚合：

```text
r_k(t) = Σ_i q_i,k(t) z_i(t) / Σ_i q_i,k(t)
```

含义：

- `z_i(t)`：节点 `i` 在时间 `t` 的动态表示。
- `q_i,k(t)`：节点 `i` 属于簇 `k` 的概率。
- 高置信属于某个簇的节点，会更多影响该簇的原型演化。

### 6.2 端到端簇分配

节点的簇分配直接由动态原型计算：

```text
q_i,k(t) = softmax(-||z_i(t) - C_k(t)||^2 / τ)
```

这样，聚类分配成为训练图的一部分，不再依赖训练后的 KMeans。

## 7. 动态簇交互矩阵 B(t)

只引入 `C(t)` 还不够，因为它主要表达“节点属于哪个簇”。但事件交互图中，一条边不一定表示两个节点属于同一簇。

尤其是 `brain` 数据，同类边比例只有约 25.3%。这说明高频连接可能发生在不同功能簇之间，而不是同簇内部。

因此需要引入：

```text
B(t) ∈ R^{K × K}
B_kl(t)：时间 t 时，簇 k 与簇 l 的交互倾向
```

对两个节点 `u` 和 `v`，簇级交互得分为：

```text
s_cluster(u, v, t) = q_u(t)^T B(t) q_v(t)
```

展开为：

```text
s_cluster(u, v, t) = Σ_k Σ_l q_u,k(t) B_kl(t) q_v,l(t)
```

如果 `u` 大概率属于簇 `k`，`v` 大概率属于簇 `l`，那么二者是否容易交互由 `B_kl(t)` 决定。

### 7.1 B(t) 与 C(t) 的区别

```text
C(t)：簇语义，回答“这个簇是什么”
B(t)：簇间关系，回答“哪些簇会发生交互”
```

二者缺一不可：

- `C(t)` 适合解释同质性强的数据，如 `school`、`dblp`、`arXivAI`。
- `B(t)` 适合解释跨簇交互强的数据，如 `brain`。
- 对所有数据，`C(t)` 提供聚类语义，`B(t)` 提供事件生成机制。

### 7.2 事件强度建模

可以将事件发生概率或强度写成：

```text
λ_uv(t) = sigmoid(
    node_affinity(z_u(t), z_v(t))
    + q_u(t)^T B(t) q_v(t)
    + time_effect(Δt)
    + node_bias(u, v)
)
```

其中：

- `node_affinity(z_u, z_v)`：节点级相似性。
- `q_u^T B q_v`：簇级交互强度。
- `time_effect(Δt)`：时间衰减或时间激发。
- `node_bias`：节点流行度或活跃度。

这比原 TGC 的结构约束更一般，因为它不强制所有交互节点都属于同一簇。

### 7.3 B(t) 的动态更新

`B(t)` 也可以分为长期项和短期项：

```text
B(t) = B^base + ΔB(t)
```

当前时间窗口中的软簇交互矩阵为：

```text
A_cluster(t) = Σ_(u,v,t) q_u(t) q_v(t)^T
```

再用记忆机制更新：

```text
h_B(t) = GRU(h_B(t-1), A_cluster(t))
B(t) = W_B h_B(t)
```

直觉是：如果某个时间窗口中簇 `k` 和簇 `l` 频繁发生事件，那么 `B_kl(t)` 会被增强。

## 8. 时间模块选择

时间模块不能盲目固定，需要结合数据特性。

### 8.1 Hawkes 过程或神经点过程

适合：

- 时间粒度细；
- 重复交互多；
- 近期事件对未来事件有强激发作用。

最适合的数据：

```text
school
```

不太适合的数据：

```text
patent、arXivAI
```

因为这些数据几乎没有重复 pair interaction，难以建模 pair-level self-excitation。

### 8.2 TGN-style Memory

TGN-style memory 更通用。它为每个节点维护一个状态：

```text
s_i(t) = MemoryUpdate(s_i(t-1), message_i(t))
z_i(t) = Encoder(s_i(t), temporal_neighbors)
```

优点：

- 能处理重复交互，也能处理非重复交互；
- 适合事件流；
- 可以给 `C(t)` 和 `B(t)` 提供节点状态输入；
- 比纯 Hawkes 更适合统一多个数据集。

### 8.3 推荐选择

推荐采用：

```text
TGN-style 节点记忆 + cluster-conditioned neural intensity
```

即：

- 用 memory 表示节点状态演化；
- 用 `C(t)` 表示动态簇语义；
- 用 `B(t)` 表示动态簇级交互；
- 用事件强度函数预测 `(u, v, t)` 是否发生。

## 9. 批处理策略

不能继续沿用原 TGC 的随机 batch 作为主训练方式。memory 或点过程模型要求时间因果性。

推荐训练流程：

```text
按时间排序事件
    -> 划分时间窗口
    -> 当前窗口预测事件
    -> 计算损失
    -> 更新模型参数
    -> 更新节点 memory、簇 memory 和 B(t) memory
```

原则：

1. 预测时间 `t` 的事件时，只能使用 `t` 之前的信息。
2. 当前 batch/window 的事件预测完成后，再更新 memory。
3. 小数据集可以使用细粒度事件流训练。
4. 大数据集使用时间窗口和负采样。

## 10. 损失函数设计

总体损失：

```text
L = L_event
  + λ1 L_proto
  + λ2 L_cluster_interaction
  + λ3 L_temporal_smooth
  + λ4 L_balance
  + λ5 L_contrast
```

### 10.1 事件预测损失

正事件强度高，负采样事件强度低：

```text
L_event = -log λ_uv(t) - Σ_neg log(1 - λ_un(t))
```

### 10.2 动态原型聚类损失

节点应靠近其软分配的动态原型：

```text
L_proto = Σ_i Σ_k q_i,k(t) ||z_i(t) - C_k(t)||^2
```

### 10.3 簇交互损失

观测事件应能被 `B(t)` 解释：

```text
L_cluster_interaction = BCE(q_u(t)^T B(t) q_v(t), y_uv)
```

### 10.4 时间平滑损失

簇原型和簇交互矩阵不能无意义剧烈漂移：

```text
L_smooth_C = Σ_k ||C_k(t) - C_k(t-1)||^2
L_smooth_B = ||B(t) - B(t-1)||_F^2
```

可使用时间间隔加权：

```text
w(Δt) = exp(-Δt / σ)
```

短时间间隔要求更平滑，长时间间隔允许更大漂移。

### 10.5 防塌缩均衡损失

防止所有节点被分到少数簇：

```text
p_k = mean_i q_i,k(t)
L_balance = KL(p || uniform)
```

### 10.6 结构对比损失

保留原 TGC 中拓扑结构建模的优点：

```text
正样本：真实时间邻居
负样本：未交互节点
```

这部分用于增强节点级判别性。

## 11. 硬件与可扩展性估计

当前 baseline 的主要资源成本包括：

1. 固定负采样表 `1e8`，约 763MB CPU 内存。
2. 每轮全量 KMeans，CPU 时间和内存开销明显。
3. 全量节点 embedding 常驻 GPU。

当前 embedding 相关内存估计：

| 数据集 | 节点张量显存估计 | KMeans CPU 输入 |
|---|---:|---:|
| patent | 约 18MB | 约 12MB |
| dblp | 约 41MB | 约 27MB |
| arXivAI | 约 103MB | 约 68MB |
| arXivPhy | 约 1.23GB | 约 818MB |
| arXivLarge | 约 1.95GB | 约 1.29GB |
| brain | 约 7MB | 约 5MB |
| school | 小于 1MB | 小于 1MB |

新方法如果加入 TGN-style node memory，至少会多一份 `N × d` 的 memory 表。对以下数据集较容易承受：

```text
patent、dblp、school、brain、arXivAI
```

对 `arXivLarge`、`arXivPhy` 需要考虑：

- 更大显存；
- CPU/offload node memory；
- 时间窗口采样；
- 邻居采样；
- 降低全量评估频率。

第一阶段建议实验数据集：

```text
patent + dblp + school + brain
```

它们分别覆盖稀疏 citation-like、同质性中等图、细粒度重复接触网络、稠密跨簇交互网络。

## 12. 实验设计

### 12.1 Baseline

建议对比：

- 原始 TGC；
- node2vec + KMeans；
- 静态图聚类方法；
- 端到端静态原型版本；
- 动态原型但不使用 `B(t)`；
- 使用学到 embedding 后再 KMeans 的版本。

### 12.2 消融实验

必须包含：

1. `Static C`：静态原型，不使用 `C(t)`。
2. `w/o B(t)`：移除动态簇交互矩阵。
3. `Static B`：使用静态 `B`，不随时间演化。
4. `w/o smoothness`：移除时间平滑约束。
5. `w/o balance`：移除防塌缩均衡损失。
6. `Post-hoc KMeans`：对学到的 embedding 再做 KMeans。
7. `Random batch vs chronological batch`：验证时间因果训练的重要性。

### 12.3 指标

常规聚类指标：

- ACC
- NMI
- ARI
- F1

动态图诊断指标：

- assignment entropy：簇分配置信度；
- cluster balance：簇规模均衡性；
- prototype drift：`C(t)` 漂移量；
- interaction drift：`B(t)` 漂移量；
- switch rate：节点簇迁移率；
- event prediction AUC/AP：事件预测能力；
- cross-cluster interaction analysis：跨簇交互解释能力。

## 13. 论文贡献表达

可以将创新点组织为：

1. 提出端到端时间图聚类框架，避免预训练 embedding + 后验 KMeans 的目标不一致。
2. 提出动态簇原型 `C(t)`，将时间信息从节点表示层提升到簇语义层。
3. 提出动态簇交互矩阵 `B(t)`，显式建模同簇和跨簇事件生成。
4. 使用时间因果的 window-based training 替代随机 batch。
5. 通过多类型数据集验证模型对稀疏 citation-like 图、高频接触图、跨簇交互图的适应性。

核心论断：

```text
Temporal graph clustering should model clusters as dynamic objects.
```

进一步展开：

```text
C(t) captures evolving cluster semantics.
B(t) captures evolving cluster-level event interactions.
```

## 14. 阶段性实现路线

建议分三阶段推进。

### 阶段一：端到端静态原型

目标：

```text
先替代 KMeans，验证端到端聚类训练有效
```

实现：

- 保留简化节点 embedding 或轻量 temporal encoder；
- 引入可学习原型 `C`；
- 输出 `q_i`；
- 加入 `L_proto` 和 `L_balance`。

### 阶段二：动态 C(t)

目标：

```text
验证簇语义随时间演化是否提升 temporal clustering
```

实现：

- 按时间窗口训练；
- 引入 cluster memory；
- 生成 `C(t)`；
- 加入 prototype drift 诊断和时间平滑损失。

### 阶段三：动态 B(t)

目标：

```text
建模跨簇事件生成，增强对 brain 等低同质性数据的解释能力
```

实现：

- 构造软簇交互矩阵 `A_cluster(t)`；
- 生成 `B(t)`；
- 在事件强度函数中加入 `q_u^T B(t) q_v`；
- 做 `w/o B(t)` 和 `Static B` 消融。

## 15. 总结

当前项目可以作为 TGC baseline，但新方法不应局限于原 TGC 的两阶段框架。

推荐技术路线是：

```text
TGN-style 节点记忆
    + 动态簇原型 C(t)
    + 动态簇交互矩阵 B(t)
    + 端到端聚类分配
    + 时间因果窗口训练
```

相比原 TGC，新路线的本质区别是：

```text
原 TGC：用时间辅助节点 embedding，再后验聚类
新方法：直接建模时间演化的簇语义和簇间交互，并端到端输出聚类结果
```

这条路线更符合事件交互图的生成机制，也能形成比“去掉 KMeans”更充分的研究创新点。
