import sys
import numpy as np
import random
sys.path.append('..')
import nrekit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import sklearn.metrics 
import copy

class Siamese(nn.Module):

    def __init__(self, sentence_encoder, hidden_size=230):
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder # Should be different from main sentence encoder
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, 1)
        self.cost = nn.BCELoss(reduction="none")
        self.drop = nn.Dropout()
        self._accuracy = 0.0

    def forward_snowball_style(self, data, positive_support_size, threshold=0.5):
        support, query, unlabelled = data

        x = self.sentence_encoder(support)[:positive_support_size]
        y = self.sentence_encoder(unlabelled)
        assert(y.size(0) == 50 * 5)
        unlabelled_size = y.size(0)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        dis = torch.pow(x - y, 2).view(-1, self.hidden_size)
        score = torch.sigmoid(self.fc(dis).squeeze())
        label = torch.zeros((positive_support_size, unlabelled_size)).long().cuda()
        label[:, :50] = 1
        label = label.view(-1)
        self._loss = self.cost(score, label.float()).mean()
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        self._accuracy = torch.mean((pred == label).type(torch.FloatTensor))
        self._prec = float(np.logical_and(pred == 1, label == 1).sum()) / float((pred == 1).sum() + 1)
        self._recall = float(np.logical_and(pred == 1, label == 1).sum()) / float((label == 1).sum() + 1)

    def forward(self, data, num_size, num_class, threshold=0.5):
        x = self.sentence_encoder(data).contiguous().view(num_class, num_size, -1)
        x1 = x[:, :num_size//2].contiguous().view(-1, self.hidden_size)
        x2 = x[:, num_size//2:].contiguous().view(-1, self.hidden_size)
        y1 = x[:num_class//2,:].contiguous().view(-1, self.hidden_size)
        y2 = x[num_class//2:,:].contiguous().view(-1, self.hidden_size)
        # y1 = x[0].contiguous().unsqueeze(0).expand(x.size(0) - 1, -1, -1).contiguous().view(-1, self.hidden_size)
        # y2 = x[1:].contiguous().view(-1, self.hidden_size)
        label = torch.zeros((x1.size(0) + y1.size(0))).long().cuda()
        label[:x1.size(0)] = 1
        z1 = torch.cat([x1, y1], 0)
        z2 = torch.cat([x2, y2], 0)
        dis = torch.pow(z1 - z2, 2)
        score = torch.sigmoid(self.fc(dis).squeeze())
        self._loss = self.cost(score, label.float()).mean()
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        self._accuracy = torch.mean((pred == label).type(torch.FloatTensor))
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        self._prec = float(np.logical_and(pred == 1, label == 1).sum()) / float((pred == 1).sum() + 1)
        self._recall = float(np.logical_and(pred == 1, label == 1).sum()) / float((label == 1).sum() + 1)

    def forward_infer(self, x, y, threshold=0.5):
        x = self.sentence_encoder(x)
        support_size = x.size(0)
        y = self.sentence_encoder(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        dis = torch.pow(x - y, 2).view(-1, self.hidden_size)
        score = torch.sigmoid(self.fc(dis).squeeze(-1))
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        pred = pred.view(support_size, -1).sum(0)
        pred[pred < 1] = 0
        pred[pred > 0] = 1
        return pred

    def forward_infer_sort(self, x, y):
        x = self.sentence_encoder(x)
        support_size = x.size(0)
        y = self.sentence_encoder(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        dis = torch.pow(x - y, 2)
        score = torch.sigmoid(self.fc(dis).squeeze(-1)).mean(0)
        pred = []
        for i in range(score.size(0)):
            pred.append((score[i], i))
        pred.sort(key=lambda x: x[0], reverse=True)
        return pred

    def forward_infer_half(self, x, y, threshold=0.5):
        x = self.sentence_encoder(x)
        support_size = x.size(0)
        y = self.sentence_encoder(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        dis = torch.pow(x - y, 2).view(-1, self.hidden_size)
        score = torch.sigmoid(self.fc(dis).squeeze(-1))
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        pred = pred.view(support_size, -1).sum(0)
        pred[pred < support_size // 2] = 0
        pred[pred > 0] = 1
        return pred
    
class Snowball(nrekit.framework.Model):
    
    def __init__(self, sentence_encoder, base_class, siamese_model, hidden_size=230, drop_rate=0.5):
        nrekit.framework.Model.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.base_class = base_class
        self.fc = nn.Linear(hidden_size, base_class)
        self.drop = nn.Dropout(drop_rate)
        self.siamese_model = siamese_model
        self.cost = nn.BCEWithLogitsLoss()
        
        # snowball hyperparameter
        self.parser.add_argument("--phase1_add_num", help="number of instances added in phase 1", type=int, default=10)
        self.parser.add_argument("--phase2_add_num", help="number of instances added in phase 2", type=int, default=10)
        self.parser.add_argument("--phase1_siamese_th", help="threshold of relation siamese network in phase 1", type=float, default=0)
        self.parser.add_argument("--phase2_siamese_th", help="threshold of relation siamese network in phase 2", type=float, default=0)
        self.parser.add_argument("--phase2_cl_th", help="threshold of relation classifier in phase 2", type=float, default=0.9)

        # fine-tune hyperparameter
        self.parser.add_argument("--finetune_epoch", help="num of epochs when finetune", type=float, default=20)
        self.parser.add_argument("--finetune_batch_size", help="batch size when finetune", type=float, default=50)
        self.parser.add_argument("--finetune_lr", help="learning rate when finetune", type=float, default=0.1)
        self.parser.add_argument("--finetune_wd", help="weight decay rate when finetune", type=float, default=1e-5)

        self.args = self.parser.parse_args()

    def __loss__(self, logits, label):
        onehot_label = torch.zeros(logits.size()).cuda()
        onehot_label.scatter_(1, label.view(-1, 1), 1)
        return self.cost(logits, onehot_label)

    def forward_base(self, data):
        batch_size = data['word'].size(0)
        x = self.sentence_encoder(data) # (batch_size, hidden_size)
        x = self.drop(x)
        x = self.fc(x) # (batch_size, base_class)

        # x = torch.sigmoid(x)
        # label = torch.zeros((batch_size, self.base_class)).cuda()
        # label.scatter_(1, data['label'].view(-1, 1), 1) # (batch_size, base_class)
        # loss_array = self.__loss__(x, label)
        # self._loss = ((label.view(-1) + 1.0 / self.base_class) * loss_array).mean() * self.base_class

        self._loss = self.__loss__(x, data['label'])
        _, pred = x.max(-1)
        self._accuracy = self.__accuracy__(pred, data['label'])
        self._pred = pred
    
    def forward_baseline(self, support_pos, support_neg, query, threshold=0.5):
        '''
        baseline model
        support_pos: positive support set
        support_neg: negative support set
        query: query set
        threshold: ins whose prob > threshold are predicted as positive
        '''
        # concat
        support = {}
        support['word'] = torch.cat([support_pos['word'], support_neg['word']], 0)
        support['pos1'] = torch.cat([support_pos['pos1'], support_neg['pos1']], 0)
        support['pos2'] = torch.cat([support_pos['pos2'], support_neg['pos2']], 0)
        support['mask'] = torch.cat([support_pos['mask'], support_neg['mask']], 0)
        support['label'] = torch.cat([support_pos['label'], support_neg['label']], 0)
        
        # train
        self._train_finetune_init()
        support_rep = self.sentence_encoder(support)
        self._train_finetune(support_rep, support['label'])
        
        # test
        query_prob = self._infer(query).cpu().detach().numpy()
        label = query['label'].cpu().detach().numpy()
        self._baseline_accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1), np.logical_and(query_prob < threshold, label == 0)).sum()) / float(query_prob.shape[0])
        if (query_prob > threshold).sum() == 0:
            self._baseline_prec = 0
        else:        
            self._baseline_prec = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((query_prob > threshold).sum())
        self._baseline_recall = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((label == 1).sum())
        if self._baseline_prec + self._baseline_recall == 0:
            self._baseline_f1 = 0
        else:
            self._baseline_f1 = float(2.0 * self._baseline_prec * self._baseline_recall) / float(self._baseline_prec + self._baseline_recall)
        self._baseline_auc = sklearn.metrics.roc_auc_score(label, query_prob)
        print('')
        sys.stdout.write('[BASELINE EVAL] acc: {0:2.2f}%, prec: {1:2.2f}%, rec: {2:2.2f}%, f1: {3:1.3f}, auc: {4:1.3f}'.format( \
            self._baseline_accuracy * 100, self._baseline_prec * 100, self._baseline_recall * 100, self._baseline_f1, self._baseline_auc))
        print('')

    def _train_finetune_init(self):
        # init variables and optimizer
        self.new_W = Variable(self.fc.weight.mean(0) / 1e3, requires_grad=True)
        self.new_bias = Variable(torch.zeros((1)), requires_grad=True)
        self.optimizer = optim.Adam([self.new_W, self.new_bias], self.args.finetune_lr, weight_decay=self.args.finetune_wd)
        self.new_W = self.new_W.cuda()
        self.new_bias = self.new_bias.cuda()

    def _train_finetune(self, data_repre, label, learning_rate=None, weight_decay=1e-5):
        '''
        train finetune classifier with given data
        data_repre: sentence representation (encoder's output)
        label: label
        '''

        optimizer = self.optimizer
        if learning_rate is not None:
            optimizer = optim.Adam([self.new_W, self.new_bias], learning_rate, weight_decay=weight_decay)

        # hyperparameters
        max_epoch = self.args.finetune_epoch
        batch_size = self.args.finetune_batch_size

        # dropout
        data_repre = self.drop(data_repre) 
        
        # train
        print('')
        for epoch in range(max_epoch):
            max_iter = data_repre.size(0) // batch_size
            if data_repre.size(0) % batch_size != 0:
                max_iter += 1
            order = list(range(data_repre.size(0)))
            random.shuffle(order)
            for i in range(max_iter):            
                x = data_repre[order[i * batch_size : min((i + 1) * batch_size, data_repre.size(0))]]
                batch_label = label[order[i * batch_size : min((i + 1) * batch_size, data_repre.size(0))]]
                x = torch.matmul(x, self.new_W) + self.new_bias # (batch_size, 1)
                x = torch.sigmoid(x)
                iter_loss = self.__loss__(x, batch_label.float()).mean()
                optimizer.zero_grad()
                iter_loss.backward(retain_graph=True)
                optimizer.step()
                sys.stdout.write('[snowball finetune] epoch {0:4} iter {1:4} | loss: {2:2.6f}'.format(epoch, i, iter_loss) + '\r')
                sys.stdout.flush()

    def _add_ins_to_data(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (list)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''
        dataset_dst['word'].append(dataset_src['word'][ins_id])
        dataset_dst['pos1'].append(dataset_src['pos1'][ins_id])
        dataset_dst['pos2'].append(dataset_src['pos2'][ins_id])
        dataset_dst['mask'].append(dataset_src['mask'][ins_id])
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'].append(dataset_src['id'][ins_id])
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'].append(label)

    def _add_ins_to_vdata(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (variable)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''
        dataset_dst['word'] = torch.cat([dataset_dst['word'], dataset_src['word'][ins_id].unsqueeze(0)], 0)
        dataset_dst['pos1'] = torch.cat([dataset_dst['pos1'], dataset_src['pos1'][ins_id].unsqueeze(0)], 0)
        dataset_dst['pos2'] = torch.cat([dataset_dst['pos2'], dataset_src['pos2'][ins_id].unsqueeze(0)], 0)
        dataset_dst['mask'] = torch.cat([dataset_dst['mask'], dataset_src['mask'][ins_id].unsqueeze(0)], 0)
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'].append(dataset_src['id'][ins_id])
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'] = torch.cat([dataset_dst['label'], torch.ones((1)).long().cuda()], 0)

    def _dataset_stack_and_cuda(self, dataset):
        '''
        stack the dataset to torch.Tensor and use cuda mode
        dataset: target dataset
        '''
        if (len(dataset['word']) == 0):
            return
        dataset['word'] = torch.stack(dataset['word'], 0).cuda()
        dataset['pos1'] = torch.stack(dataset['pos1'], 0).cuda()
        dataset['pos2'] = torch.stack(dataset['pos2'], 0).cuda()
        dataset['mask'] = torch.stack(dataset['mask'], 0).cuda()

    def _infer(self, dataset):
        '''
        get prob output of the finetune network with the input dataset
        dataset: input dataset
        return: prob output of the finetune network
        '''
        x = self.sentence_encoder(dataset)
        x = torch.matmul(x, self.new_W) + self.new_bias # (batch_size, 1)
        x = torch.sigmoid(x)
        return x.view(-1)

    def _forward_train(self, support_pos, support_neg, query, distant, threshold=0.5):
        '''
        snowball process (train)
        support_pos: support set (positive, raw data)
        support_neg: support set (negative, raw data)
        query: query set
        distant: distant data loader
        threshold: ins with prob > threshold will be classified as positive
        threshold_for_phase1: distant ins with prob > th_for_phase1 will be added to extended support set at phase1
        threshold_for_phase2: distant ins with prob > th_for_phase2 will be added to extended support set at phase2
        '''

        # hyperparameters
        snowball_max_iter = 2
        sys.stdout.flush()
        candidate_num_class = 20
        candidate_num_ins_per_class = 100
        
        sort_num1 = self.args.phase1_add_num
        sort_num2 = self.args.phase2_add_num
        sort_threshold1 = self.args.phase1_siamese_th
        sort_threshold2 = self.args.phase2_siamese_th
        sort_ori_threshold = self.args.phase2_cl_th

        # get neg representations with sentence encoder
        support_neg_rep = self.sentence_encoder(support_neg)
        
        # init
        self._train_finetune_init()
        self._metric = []

        # copy
        original_support_pos = copy.deepcopy(support_pos)

        # snowball
        exist_id = {}
        print('\n-------------------------------------------------------')
        for snowball_iter in range(snowball_max_iter):
            print('###### snowball iter ' + str(snowball_iter))
            # phase 1: expand positive support set from distant dataset (with same entity pairs)

            ## get all entpairs and their ins in positive support set
            old_support_pos_label = support_pos['label'] + 0
            entpair_support = {}
            entpair_distant = {}
            for i in range(len(support_pos['id'])): # only positive support
                entpair = support_pos['entpair'][i]
                exist_id[support_pos['id'][i]] = 1
                if entpair not in entpair_support:
                    entpair_support[entpair] = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
                self._add_ins_to_data(entpair_support[entpair], support_pos, i)
            
            ## pick all ins with the same entpairs in distant data and choose with siamese network
            self._phase1_add_num = 0 # total number of snowball instances
            self._phase1_total = 0
            for entpair in entpair_support:
                raw = distant.get_same_entpair_ins(entpair) # ins with the same entpair
                if raw is None:
                    continue
                entpair_distant[entpair] = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
                for i in range(raw['word'].size(0)):
                    if raw['id'][i] not in exist_id: # don't pick sentences already in the support set
                        self._add_ins_to_data(entpair_distant[entpair], raw, i)
                self._dataset_stack_and_cuda(entpair_support[entpair])
                self._dataset_stack_and_cuda(entpair_distant[entpair])
                if len(entpair_support[entpair]['word']) == 0 or len(entpair_distant[entpair]['word']) == 0:
                    continue
                pick_or_not = self.siamese_model.forward_infer_sort(entpair_support[entpair], entpair_distant[entpair])
                # pick_or_not = self.siamese_model.forward_infer_sort(original_support_pos, entpair_distant[entpair], threshold=threshold_for_phase1)
                # pick_or_not = self._infer(entpair_distant[entpair]) > threshold
      
                # -- method A: use threshold --
                '''
                for i in range(pick_or_not.size(0)):
                    if pick_or_not[i]:
                        self._add_ins_to_vdata(support_pos, entpair_distant[entpair], i, label=1)
                        exist_id[entpair_distant[entpair]['id'][i]] = 1
                self._phase1_add_num += pick_or_not.sum()
                self._phase1_total += pick_or_not.size(0)
                '''
                
                # -- method B: use sort --
                for i in range(min(len(pick_or_not), sort_num1)):
                    if pick_or_not[i][0] > sort_threshold1:
                        iid = pick_or_not[i][1]
                        self._add_ins_to_vdata(support_pos, entpair_distant[entpair], iid, label=1)
                        exist_id[entpair_distant[entpair]['id'][iid]] = 1
                        self._phase1_add_num += 1
                self._phase1_total += entpair_distant[entpair]['word'].size(0)

            ## build new support set
            support_pos_rep = self.sentence_encoder(support_pos)
            support_rep = torch.cat([support_pos_rep, support_neg_rep], 0)
            support_label = torch.cat([support_pos['label'], support_neg['label']], 0)
            
            ## finetune
            # self._train_finetune_init()
            self._train_finetune(support_rep, support_label)
            self._forward_eval_binary(query, threshold)
            self._metric.append(np.array([self._f1, self._prec, self._recall]))
            print('\nphase1 add {} ins / {}'.format(self._phase1_add_num, self._phase1_total))

            # phase 2: use the new classifier to pick more extended support ins
            self._phase2_add_num = 0
            candidate = distant.get_random_candidate(self.pos_class, candidate_num_class, candidate_num_ins_per_class)

            ## -- method 1: directly use the classifier --
            candidate_prob = self._infer(candidate)
            ## -- method 2: use siamese network --
            pick_or_not = self.siamese_model.forward_infer_sort(support_pos, candidate)

            ## -- method A: use threshold --
            '''
            self._phase2_total = candidate_prob.size(0)
            for i in range(candidate_prob.size(0)):
                # if (candidate_prob[i] > threshold_for_phase2) and not (candidate['id'][i] in exist_id):
                if (pick_or_not[i]) and (candidate_prob[i] > threshold_for_phase2) and not (candidate['id'][i] in exist_id):
                    exist_id[candidate['id'][i]] = 1 
                    self._phase2_add_num += 1
                    self._add_ins_to_vdata(support_pos, candidate, i, label=1)
            '''

            ## -- method B: use sort --
            self._phase2_total = candidate['word'].size(0)
            for i in range(min(len(candidate_prob), sort_num2)):
                iid = pick_or_not[i][1]
                if (pick_or_not[i][0] > sort_threshold2) and (candidate_prob[iid] > sort_ori_threshold) and not (candidate['id'][iid] in exist_id):
                    exist_id[candidate['id'][iid]] = 1 
                    self._phase2_add_num += 1
                    self._add_ins_to_vdata(support_pos, candidate, iid, label=1)

            ## build new support set
            support_pos_rep = self.sentence_encoder(support_pos)
            support_rep = torch.cat([support_pos_rep, support_neg_rep], 0)
            support_label = torch.cat([support_pos['label'], support_neg['label']], 0)

            ## finetune
            self._train_finetune(support_rep, support_label)
            self._forward_eval_binary(query, threshold)
            self._metric.append(np.array([self._f1, self._prec, self._recall]))
            print('\nphase2 add {} ins / {}'.format(self._phase2_add_num, self._phase2_total))

    def _forward_eval_binary(self, query, threshold=0.5):
        '''
        snowball process (eval)
        query: query set (raw data)
        threshold: ins with prob > threshold will be classified as positive
        return (accuracy at threshold, precision at threshold, recall at threshold, f1 at threshold, auc), 
        '''
        query_prob = self._infer(query).cpu().detach().numpy()
        label = query['label'].cpu().detach().numpy()
        accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1), np.logical_and(query_prob < threshold, label == 0)).sum()) / float(query_prob.shape[0])
        if (query_prob > threshold).sum() == 0:
            precision = 0
        else:
            precision = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((query_prob > threshold).sum())
        recall = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((label == 1).sum())
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = float(2.0 * precision * recall) / float(precision + recall)
        auc = sklearn.metrics.roc_auc_score(label, query_prob)
        print('')
        sys.stdout.write('[EVAL] acc: {0:2.2f}%, prec: {1:2.2f}%, rec: {2:2.2f}%, f1: {3:1.3f}, auc: {4:1.3f}'.format(\
                accuracy * 100, precision * 100, recall * 100, f1, auc) + '\r')
        sys.stdout.flush()
        self._accuracy = accuracy
        self._prec = precision
        self._recall = recall
        self._f1 = f1
        return (accuracy, precision, recall, f1, auc)

    def forward(self, support_pos, support_neg, query, distant, pos_class, threshold=0.5, threshold_for_snowball=0.5):
        '''
        snowball process (train + eval)
        support_pos: support set (positive, raw data)
        support_neg: support set (negative, raw data)
        query: query set (raw data)
        distant: distant data loader
        pos_class: positive relation (name)
        threshold: ins with prob > threshold will be classified as positive
        threshold_for_snowball: distant ins with prob > th_for_snowball will be added to extended support set
        '''
        self.pos_class = pos_class 

        self._forward_train(support_pos, support_neg, query, distant, threshold=threshold)

    def init_10shot(self, Ws, bs):
        self.Ws = torch.stack(Ws, 0).transpose(0, 1) # (230, 16)
        self.bs = torch.stack(bs, 0).transpose(0, 1) # (1, 16)

    def eval_10shot(self, query):
        x = self.sentence_encoder(query)
        x = torch.matmul(x, self.Ws) + self.new_bias # (batch_size, 16)
        x = torch.sigmoid(x)
        _, pred = x.max(-1) # (batch_size)
        return self.__accuracy__(pred, query['label'])

