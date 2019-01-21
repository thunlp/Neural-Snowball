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

class Siamese(nn.Module):

    def __init__(self, sentence_encoder, hidden_size=230):
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder # Should be different from main sentence encoder
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, 1)
        self.cost = nn.BCELoss(reduction="none")
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
        score = F.sigmoid(self.fc(dis).squeeze())
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
        x1 = x[:, :num_size/2].contiguous().view(-1, self.hidden_size)
        x2 = x[:, num_size/2:].contiguous().view(-1, self.hidden_size)
        y1 = x[:num_class/2,:].contiguous().view(-1, self.hidden_size)
        y2 = x[num_class/2:,:].contiguous().view(-1, self.hidden_size)
        # y1 = x[0].contiguous().unsqueeze(0).expand(x.size(0) - 1, -1, -1).contiguous().view(-1, self.hidden_size)
        # y2 = x[1:].contiguous().view(-1, self.hidden_size)
        label = torch.zeros((x1.size(0) + y1.size(0))).long().cuda()
        label[:x1.size(0)] = 1
        z1 = torch.cat([x1, y1], 0)
        z2 = torch.cat([x2, y2], 0)
        dis = torch.pow(z1 - z2, 2)
        score = F.sigmoid(self.fc(dis).squeeze())
        self._loss = self.cost(score, label.float()).mean()
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        self._accuracy = torch.mean((pred == label).type(torch.FloatTensor))
        self._prec = float(np.logical_and(pred == 1, label == 1).sum()) / float((pred == 1).sum() + 1)
        self._recall = float(np.logical_and(pred == 1, label == 1).sum()) / float((label == 1).sum() + 1)

    def forward_infer(self, x, y, threshold=0.5):
        x = self.sentence_encoder(x)
        support_size = x.size(0)
        y = self.sentence_encoder(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        dis = torch.pow(x - y, 2).view(-1, self.hidden_size)
        score = F.sigmoid(self.fc(dis).squeeze(-1))
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        pred = pred.view(support_size, -1).sum(0)
        pred[pred >= 1] = 1
        return pred
    
class Snowball(nrekit.framework.Model):
    
    def __init__(self, sentence_encoder, base_class, siamese_model, hidden_size=230):
        nrekit.framework.Model.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.base_class = base_class
        self.fc = nn.Linear(hidden_size, base_class)
        self.drop = nn.Dropout()
        self.siamese_model = siamese_model
        self.cost = nn.BCELoss(reduction='none')

    def forward_base(self, data):
        batch_size = data['word'].size(0)
        x = self.sentence_encoder(data) # (batch_size, hidden_size)
        x = self.drop(x)
        x = self.fc(x) # (batch_size, base_class)
        x = F.sigmoid(x)
        label = torch.zeros((batch_size, self.base_class)).cuda()
        label.scatter_(1, data['label'].view(-1, 1), 1) # (batch_size, base_class)
        loss_array = self.__loss__(x, label)
        self._loss = ((label.view(-1) + 1.0 / self.base_class) * loss_array).mean() * self.base_class
        _, pred = x.max(-1)
        self._accuracy = self.__accuracy__(pred, data['label'])

    def forward_new(self, data, positive_support_size, threshold=0.5):
        support, query, unlabelled = data
        new_W = Variable(self.fc.weight.mean(0) / 1e3, requires_grad=True)
        new_bias = Variable(torch.zeros((1)), requires_grad=True)
        optimizer = optim.Adam([new_W, new_bias], 1e-1, weight_decay=0)
        new_W = new_W.cuda()
        new_bias = new_bias.cuda()

        # Expand
        support_x = self.sentence_encoder(support) # (batch_size, hidden_size)
        unlabelled_x = self.sentence_encoder(unlabelled)
        similarity = self.siamese_model.forward_infer(support, unlabelled, support_size=positive_support_size, threshold=0.95)
        chosen = []
        correct_snowball = 0
        assert(similarity.size(0) == 50 * 5)
        for i in range(similarity.size(0)):
            if similarity[i] == 1:
                chosen.append(unlabelled_x[i])
                if i <= 50:
                    correct_snowball += 1
        self._correct_snowball = correct_snowball
        self._snowball = similarity.sum()
        if similarity.sum() > 0:
            chosen = torch.stack(chosen, 0)
            chosen = torch.cat([support_x, chosen], 0)
            label = torch.cat([support['label'], torch.ones((chosen.size(0) - support['label'].size(0))).long().cuda()], 0)
        else:
            chosen = support_x
            label = support['label']

        '''
        for i in range(10):
            x = torch.matmul(support_x, new_W) + new_bias # (batch_size, 1)
            x = F.sigmoid(x)
            iter_loss_array = self.__loss__(x, support['label'].float())
            iter_loss = iter_loss_array.mean()
            iter_loss.backward(retain_graph=True)
            optimizer.step()
        '''

        for i in range(10):
            x = torch.matmul(chosen, new_W) + new_bias # (batch_size, 1)
            x = F.sigmoid(x)
            iter_loss_array = self.__loss__(x, label.float())
            iter_loss = iter_loss_array.mean()
            optimizer.zero_grad()
            iter_loss.backward(retain_graph=True)
            optimizer.step()
        
        # Test
        query_x = self.sentence_encoder(query) # (batch_size, hidden_size)
        query_x = self.drop(query_x)
        x = torch.matmul(query_x, new_W) + new_bias # (batch_size, 1)
        x = F.sigmoid(x)
        loss_array = self.__loss__(x, query['label'].float())
        self._loss = loss_array.mean()
        pred = torch.zeros((x.size(0))).long().cuda()
        pred[x > threshold] = 1
        self._accuracy = self.__accuracy__(pred, query['label'])
        pred = pred.view(-1).data.cpu().numpy()
        label = query['label'].view(-1).data.cpu().numpy()
        self._prec = float(np.logical_and(pred == 1, label == 1).sum()) / float((pred == 1).sum() + 1)
        self._recall = float(np.logical_and(pred == 1, label == 1).sum()) / float((label == 1).sum() + 1)

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
        query_prob = self._infer(query)
        self._baseline_accuracy = float((query_prob > threshold).sum()) / float(query_prob.shape[0])
        self._baseline_prec = float(np.logical_and(query_prob > threshold, query['label'] == 1).sum()) / float((query_prob > threshold).sum())
        self._baseline_recall = float(np.logical_and(query_prob > threshold, query['label'] == 1).sum()) / float((query['label'] == 1).sum())
        self._baseline_f1 = 2 * precision * recall / (precision + recall)
        self._baseline_auc = sklearn.metrics.roc_auc_score(query['label'], query_prob)

    def _train_finetune_init(self):
        # init variables and optimizer
        self.new_W = Variable(self.fc.weight.mean(0) / 1e3, requires_grad=True)
        self.new_bias = Variable(torch.zeros((1)), requires_grad=True)
        self.optimizer = optim.Adam([new_W, new_bias], 1e-1, weight_decay=1e-5)
        self.new_W = new_W.cuda()
        self.new_bias = new_bias.cuda()

    def _train_finetune(self, data_repre, label):
        '''
        train finetune classifier with given data
        data_repre: sentence representation (encoder's output)
        label: label
        '''
        
        # hyperparameters
        max_epoch = 20
        batch_size = 50

        # dropout
        data_repre = self.drop(data_repre) 
        
        # train
        print('')
        for epoch in range(max_epoch):
            max_iter = data_repre.size(0) # batch_size
            if data_repre.size(0) % batch_size != 0:
                max_iter += 1
            order = list(range(data_repre.size(0)))
            random.shuffle(order)
            for i in range(max_iter):            
                x = data_repre[order[i * batch_size, min((i + 1) * batch_size, data_repre.size(0))]]
                x = torch.matmul(x, self.new_W) + self.new_bias # (batch_size, 1)
                x = F.sigmoid(x)
                iter_loss = self.__loss__(x, label.float()).mean()
                self.optimizer.zero_grad()
                iter_loss.backward(retain_graph=True)
                self.optimizer.step()
                sys.stdout.write('[snowball finetune] epoch {0:4} iter {1:4} | loss: {2:2.6f}'.format(epoch, i, iter_loss) + '\r')
                sys.stdout.flush()

    def _add_ins_to_data(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst
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

    def _dataset_stack_and_cuda(self, dataset):
        '''
        stack the dataset to torch.Tensor and use cuda mode
        dataset: target dataset
        '''
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
        x = F.sigmoid(x)
        return x.view(-1)

    def _forward_train(self, support_pos, support_neg, query, distant, threshold=0.5, threshold_for_phase1=0.8, threshold_for_phase2=0.9):
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
        snowball_max_iter = 1
        sys.stdout.flush()
        candidate_num_class = 20
        candidate_num_ins_per_class = 100

        # get neg representations with sentence encoder
        support_neg_rep = self.sentence_encoder(support_neg)
        
        # init
        self._train_finetune_init()

        # snowball
        exist_id = {}
        for snowball_iter in range(snowball_max_iter):
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
                pick_or_not = self.siamese_model.forward_infer(entpair_support[entpair], entpair_distant[entpair], threshold=threshold_for_phase1)
      
                for i in range(pick_or_not.size(0)):
                    if pick_or_not[i] == 1:
                        self._add_ins_to_data(support_pos, entpair_distant[entpair], i, label=1)
                        exist_id[entpair_distant[entpair]['id'][i]] = 1
                self._phase1_add_num += pick_or_not.sum()
            
            ## build new support set
            support_pos_rep = self.sentence_encoder(support_pos)
            support_rep = torch.cat([support_pos_rep, support_neg_rep], 0)
            support_label = torch.cat([support_pos['label'], support_neg['label']], 0)
            
            ## finetune
            self._train_finetune(support_rep, support_label)
            self._forward_eval_binary(query, threshold)

            # phase 2: use the new classifier to pick more extended support ins
            self._phase2_add_num = 0
            candidate = distant.get_random_candidate(self.pos_class, candidate_num_class, candidate_num_ins_per_class)
            candidate_prob = self._infer(candidate)
            for i in range(candidate_prob.size(0)):
                if (candidate_prob[i] > threshold_for_phase2) and (not candidate['id'][i] in exist_id):
                    exist_id[candidate['id'][i]] = 1 
                    self._phase2_add_num += 1
                    self._add_ins_to_data(support_pos, candidate, i, label=1)

            ## build new support set
            support_pos_rep = self.sentence_encoder(support_pos)
            support_rep = torch.cat([support_pos_rep, support_neg_rep], 0)
            support_label = torch.cat([support_pos['label'], support_neg['label']], 0)

            ## finetune
            self._train_finetune(support_rep, support_label)
            self._forward_eval_binary(query, threshold)

    def _forward_eval_binary(self, query, threshold=0.5)
        '''
        snowball process (eval)
        query: query set (raw data)
        threshold: ins with prob > threshold will be classified as positive
        return (accuracy at threshold, precision at threshold, recall at threshold, f1 at threshold, auc), 
        '''
        query_prob = self._infer(query)
        accuracy = float((query_prob > threshold).sum()) / float(query_prob.shape[0])
        precision = float(np.logical_and(query_prob > threshold, query['label'] == 1).sum()) / float((query_prob > threshold).sum())
        recall = float(np.logical_and(query_prob > threshold, query['label'] == 1).sum()) / float((query['label'] == 1).sum())
        f1 = 2 * precision * recall / (precision + recall)
        auc = sklearn.metrics.roc_auc_score(query['label'], query_prob)
        print('')
        sys.stdout.write('[EVAL] acc: {0:2.1f}%, prec: {1:2.1f}%, rec: {2:2.1f}%, f1: {3:2.1f}, auc: {4:2.1f}'.format(\
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
        self._forward_train(support_pos, support_neg, distant, threshold=threshold, threshold_for_phase1=threshold_for_snowball, threshold_for_phase2=threshold_for_snowball)
        self._forward_eval(query, threshold=threshold)
