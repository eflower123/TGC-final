import math

import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
import numpy as np
import sys
from model.DataSet import TGCDataSet
from model.evaluation import eva_with_diagnostics
from torch.nn import Linear
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import time
import datetime
# 强制禁用共享显存，只使用物理显存
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_FW_DIR = os.path.dirname(_MODEL_DIR)
_ROOT_DIR = os.path.dirname(_FW_DIR)


class TGC:
    def __init__(self, args):
        self.args = args
        self.the_data = args.dataset
        self.file_path = '../data/%s/%s.txt' % (self.the_data, self.the_data)
        self.emb_path = '../emb/%s/%s_TGC_%d.emb'
        self.feature_path = './pretrain/%s_feature.emb' % self.the_data
        self.label_path = '../data/%s/node2label.txt' % self.the_data
        self.labels = self.read_label()
        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.clusters
        self.save_step = args.save_step
        self.epochs = args.epoch
        self.best_acc = 0
        self.best_nmi = 0
        self.best_ari = 0
        self.best_f1 = 0
        self.best_epoch = 0

        self.data = TGCDataSet(self.file_path, self.neg_size, self.hist_len, self.feature_path, args.directed)
        self.node_dim = self.data.get_node_dim()
        self.edge_num = self.data.get_edge_num()
        self.feature = self.data.get_feature()

        self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=True)
        self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=False)
        self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

        self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cuda(), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        kmeans = KMeans(n_clusters=self.clusters, n_init=20)
        _ = kmeans.fit_predict(self.feature)
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

        self.v = 1.0
        self.batch_weight = math.ceil(self.batch / self.edge_num)

        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb, self.delta, self.cluster_layer])
        self.loss = torch.FloatTensor()
        self.scaler = GradScaler()  # 仅在CUDA可用时生效，CPU下自动禁用

        self.use_time_loss = not getattr(args, 'no_time_loss', False)
        self.use_node_loss = not getattr(args, 'no_node_loss', False)
        if getattr(args, 'no_batch_loss', False):
            self.use_res_st = False
            self.use_res_sh = False
            self.use_res_sn = False
        else:
            self.use_res_st = not getattr(args, 'no_res_st', False)
            self.use_res_sh = not getattr(args, 'no_res_sh', False)
            self.use_res_sn = not getattr(args, 'no_res_sn', False)

        self._loss_components = {}
        self._loss_sum = {}
        self._loss_batches = 0
        self._train_start_time = None

    def read_label(self):
        n2l = dict()
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                n2l[n_id] = l_id
        reader.close()
        for i in range(len(n2l)):
            labels.append(int(n2l[i]))
        return labels

    def kl_loss(self, z, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')  # l_clu
        return the_kl_loss

    def target_dis(self, emb):
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        tmp_q = q.data
        weight = tmp_q ** 2 / tmp_q.sum(0)
        p = (weight.t() / weight.sum(1)).t()

        return p

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        s_p = self.target_dis(s_pre_emb)
        s_kl_loss = self.kl_loss(s_node_emb, s_p)
        l_node = s_kl_loss

        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)  # [b]
        res_st_loss = torch.norm(1 - new_st_adj, p=2, dim=0)
        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)  # [b,h]
        new_sh_adj = new_sh_adj * h_time_mask
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)  # [b,n]
        res_sh_loss = torch.norm(1 - new_sh_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        res_sn_loss = torch.norm(0 - new_sn_adj, p=2, dim=0).sum(dim=0, keepdims=False)

        l_node_opt = l_node * (1 if self.use_node_loss else 0)
        l_batch = res_st_loss + res_sh_loss + res_sn_loss
        l_batch_opt = (res_st_loss * (1 if self.use_res_st else 0) +
                       res_sh_loss * (1 if self.use_res_sh else 0) +
                       res_sn_loss * (1 if self.use_res_sn else 0))

        l_framework = l_node_opt + l_batch_opt

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(
            dim=1)  # [b]

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)

        total_loss = loss.sum() + l_framework if self.use_time_loss else l_framework

        self._loss_components = {
            'time_loss': loss.sum().detach(),
            'node_loss': l_node.detach(),
            'batch_loss': l_batch.detach(),
            'res_st': res_st_loss.detach(),
            'res_sh': res_sh_loss.detach(),
            'res_sn': res_sn_loss.detach(),
        }

        return total_loss

    def _acc_loss(self):
        for k in ['time_loss', 'node_loss', 'batch_loss', 'res_st', 'res_sh', 'res_sn']:
            v = self._loss_components.get(k, 0.0)
            self._loss_sum[k] += v.item() if torch.is_tensor(v) else v
        self._loss_batches += 1

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                with autocast():
                    loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                self.loss += loss.data
                self._acc_loss()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
        else:
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.data
            self._acc_loss()
            loss.backward()
            self.opt.step()

    def train(self):
        total_start = time.time()
        self._train_start_time = datetime.datetime.now()
        self._timestamp = self._train_start_time.strftime('%m%d%H')
        epoch_records = []
        prev_cluster_id = None
        prev_centers = None

        for epoch in range(self.epochs):
            self.loss = 0.0
            self._loss_sum = {k: 0.0 for k in ['time_loss', 'node_loss', 'batch_loss',
                                                 'res_st', 'res_sh', 'res_sn']}
            self._loss_batches = 0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=0)

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                self.update(sample_batched['source_node'].type(LType).cuda(),
                            sample_batched['target_node'].type(LType).cuda(),
                            sample_batched['target_time'].type(FType).cuda(),
                            sample_batched['neg_nodes'].type(LType).cuda(),
                            sample_batched['history_nodes'].type(LType).cuda(),
                            sample_batched['history_times'].type(FType).cuda(),
                            sample_batched['history_masks'].type(FType).cuda())

            avg_loss = self.loss.cpu().numpy() / len(self.data)
            n_data = len(self.data)
            num_batches = max(1, self._loss_batches)

            if self.the_data == 'arxivLarge' or self.the_data == 'arxivPhy' or self.the_data == 'arxivMath':
                labels_for_eval = None
            else:
                labels_for_eval = self.labels
            acc, nmi, ari, f1, diag, cluster_id, centers = eva_with_diagnostics(
                self.clusters, labels_for_eval, self.node_emb,
                prev_cluster_id=prev_cluster_id, prev_centers=prev_centers,
                sample_size=3000, random_state=42)
            prev_cluster_id = cluster_id
            prev_centers = centers

            if nmi > self.best_nmi and epoch > 10:
                self.best_acc = acc
                self.best_nmi = nmi
                self.best_ari = ari
                self.best_f1 = f1
                self.best_epoch = epoch
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, self.epochs))

            epoch_records.append({
                'epoch': epoch,
                'total_loss': float(avg_loss),
                'time_loss': self._loss_sum['time_loss'] / n_data,
                'node_loss': self._loss_sum['node_loss'] / num_batches,
                'batch_loss': self._loss_sum['batch_loss'] / num_batches,
                'res_st': self._loss_sum['res_st'] / num_batches,
                'res_sh': self._loss_sum['res_sh'] / num_batches,
                'res_sn': self._loss_sum['res_sn'] / num_batches,
                'acc': acc, 'nmi': nmi, 'ari': ari, 'f1': f1,
                'silhouette': diag['silhouette'],
                'dbi': diag['dbi'],
                'ch': diag['ch'],
                'empty_clusters': diag['empty_clusters'],
                'max_cluster_ratio': diag['max_cluster_ratio'],
                'cluster_size_cv': diag['cluster_size_cv'],
                'epoch_nmi': diag['epoch_nmi'],
                'switch_rate': diag['switch_rate'],
                'center_shift': diag['center_shift'],
            })

            sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, avg_loss))
            sys.stdout.write(
                'ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f) | '
                'Sil(%.4f) DBI(%.4f) CH(%.2f) | '
                'Empty(%d) MaxR(%.4f) ENMI(%.4f) Sw(%.4f)\n'
                % (acc, nmi, ari, f1,
                   diag['silhouette'], diag['dbi'], diag['ch'],
                   diag['empty_clusters'], diag['max_cluster_ratio'],
                   diag['epoch_nmi'], diag['switch_rate']))
            sys.stdout.flush()

        total_time = time.time() - total_start
        self._save_training_log(epoch_records, total_time)

        print('Best performance: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
              (self.best_acc, self.best_nmi, self.best_ari, self.best_f1))

    def _save_training_log(self, epoch_records, total_time_sec):
        out_path = os.path.join(_ROOT_DIR, 'all_results/%s/%s_TGC_%d_%s_all.txt' %
                                (self.the_data, self.the_data, self.epochs, self._timestamp))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        def fmt_time(sec):
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            return '%d:%02d:%02d' % (h, m, s)

        lines = []
        lines.append('=' * 112)
        lines.append('  TGC Training Log')
        lines.append('=' * 112)
        lines.append('  Start Time: %s' % self._train_start_time.strftime('%Y-%m-%d %H:%M'))
        lines.append('')
        lines.append('[Configuration]')
        cfg = [
            ('Dataset', self.the_data),
            ('Clusters', self.clusters),
            ('Epochs', self.epochs),
            ('Learning Rate', self.args.learning_rate),
            ('Embedding Size', self.emb_size),
            ('Batch Size', self.batch),
            ('Negative Size', self.neg_size),
            ('History Length', self.hist_len),
            ('Directed', self.args.directed),
            ('Time Loss', self.use_time_loss),
            ('Node Loss', self.use_node_loss),
            ('Batch Loss', self.use_res_st and self.use_res_sh and self.use_res_sn),
            ('  ResST', self.use_res_st),
            ('  ResSH', self.use_res_sh),
            ('  ResSN', self.use_res_sn),
        ]
        for k, v in cfg:
            lines.append('  %-18s %s' % (k + ':', v))
        lines.append('')
        lines.append('[Per-Epoch Results]')
        header = ('Epoch | TotalLoss | TimeLoss | NodeLoss | BatchLoss |'
                  '  ResST |  ResSH |  ResSN |'
                  '    ACC |    NMI |    ARI |     F1 |'
                  '     Sil |    DBI |       CH | Empty | MaxRatio | SizeCV |'
                  ' EpochNMI | SwitchRate | CenterShift')
        lines.append(header)
        sep = ('------|-----------|----------|----------|-----------|'
               '--------|--------|--------|'
               '--------|--------|--------|--------|'
               '---------|--------|----------|-------|----------|--------|'
               '----------|------------|------------')
        lines.append(sep)
        for r in epoch_records:
            line = (
                '%5d | %9.4f | %8.4f | %8.4f | %9.4f |'
                '%7.4f | %6.4f | %6.4f |'
                '%7.4f | %6.4f | %6.4f | %6.4f |'
                '%8.4f | %6.4f | %9.2f |'
                '%6d | %8.4f | %6.4f |'
                '%9.4f | %11.4f | %11.4f'
            ) % (
                r['epoch'], r['total_loss'], r['time_loss'], r['node_loss'],
                r['batch_loss'],
                r['res_st'], r['res_sh'], r['res_sn'],
                r['acc'], r['nmi'], r['ari'], r['f1'],
                r['silhouette'], r['dbi'], r['ch'],
                r['empty_clusters'], r['max_cluster_ratio'], r['cluster_size_cv'],
                r['epoch_nmi'], r['switch_rate'], r['center_shift'],
            )
            lines.append(line)
        lines.append('')
        lines.append('[Summary]')
        lines.append('  Best ACC:     %.4f (epoch %d)' % (self.best_acc, self.best_epoch))
        lines.append('  Best NMI:     %.4f (epoch %d)' % (self.best_nmi, self.best_epoch))
        lines.append('  Best ARI:     %.4f (epoch %d)' % (self.best_ari, self.best_epoch))
        lines.append('  Best F1:      %.4f (epoch %d)' % (self.best_f1, self.best_epoch))
        lines.append('  Total Time:   %s' % fmt_time(total_time_sec))
        lines.append('=' * 112)

        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('Training log saved to: %s' % out_path)

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()
