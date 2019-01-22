import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder
        self.cost = nn.BCELoss(size_average=True)
        self._loss = 0
        self._accuracy = 0
    
    def forward_base(self, data):
        raise NotImplementedError

    def forward(self, data):
        raise NotImplementedError

    def __loss__(self, logits, label):
        return self.cost(logits.view(-1), label.view(-1))

    def __accuracy__(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
    
    def loss(self):
        return self._loss
    
    def accuracy(self):
        return self._accuracy
    
class Framework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, distant):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.distant = distant 
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        if int(torch.__version__.split('.')[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              batch_size=500,
              ckpt_dir='./checkpoint',
              test_result_dir='./test_result',
              learning_rate=1.,
              lr_step_size=10000,
              weight_decay=1e-5,
              train_iter=100000,
              val_iter=100,
              val_step=1000,
              test_iter=3000,
              cuda=True,
              pretrain_model=None,
              optimizer=optim.SGD,
              model2=None):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")
        
        # Init
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        opt = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_step_size)
        if model2 is not None:
            parameters_to_optimize2 = filter(lambda x:x.requires_grad, model2.parameters())
            opt2 = optimizer(parameters_to_optimize2, learning_rate, weight_decay=weight_decay)
            scheduler2 = optim.lr_scheduler.StepLR(opt2, step_size=lr_step_size)

        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        if cuda:
            model = model.cuda()
            model2 = model2.cuda()
        # model.train()

        # Training
        best_acc = 0
        not_best_count = 0 # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        iter_loss2 = 0.0
        iter_right2 = 0.0
        iter_sample2 = 0.0

        s_num_size = 10
        s_num_class = 50

        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()
            batch_data = self.train_data_loader.next_batch(batch_size)
            model.forward_base(batch_data)
            loss = model.loss()
            right = model.accuracy()
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1

            if (model2 is not None):
                scheduler2.step()

                batch_data = self.train_data_loader.next_multi_class(num_size=s_num_size, num_class=s_num_class)
                model2(batch_data, s_num_size, s_num_class)

                loss2 = model2._loss
                right2 = model2._accuracy
                opt2.zero_grad()
                loss2.backward()
                opt2.step()

                iter_loss2 += self.item(loss2.data)
                iter_right2 += self.item(right2.data)
                iter_sample2 += 1
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}% | loss2: {3:2.6f}, accuracy2: {4:3.2f}%, prec: {5:3.2f}%, recall: {6:3.2f}%'.format( \
                    it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample, \
                    iter_loss2 / iter_sample2, 100 * iter_right2 / iter_sample2, 100.0 * model2._prec, 100.0 * model2._recall) +'\r')
                sys.stdout.flush()
            else:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) +'\r')
                sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                print("\n----------------- siamese test -------------------")
                self.eval(model2, eval_iter=val_iter, is_model2=True, threshold=0.5)
                self.eval(model2, eval_iter=val_iter, is_model2=True, threshold=0.9)
                self.eval(model2, eval_iter=val_iter, is_model2=True, threshold=0.95)
                print("\n----------------- snowball test -------------------")
                acc = self.eval(model, eval_iter=10, threshold_for_snowball=0.7)
                acc = self.eval(model, eval_iter=10, threshold_for_snowball=0.9)
                acc = self.eval(model, eval_iter=10, threshold_for_snowball=0.95)

                print("\n----------------- test end -------------------\n")
                if acc > best_acc:
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_acc = acc
                
        print("\n####################\n")
        print("Finish training " + model_name)
        test_acc = self.eval(model, ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'), eval_iter=test_iter)
        print("Test accuracy: {}".format(test_acc))

    def eval(self,
            model,
            support_size=10, query_size=10, unlabelled_size=50, query_class=10,
            s_num_size=10, s_num_class=10,
            eval_iter=10,
            ckpt=None,
            is_model2=False,
            threshold=0.5,
            threshold_for_snowball=0.5):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        # model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test_data_loader
        eval_distant_dataset = self.distant

        iter_right = 0.0
        iter_prec = 0.0
        iter_recall = 0.0
        iter_sample = 0.0
        iter_bright = 0.0
        iter_bprec = 0.0
        iter_brecall = 0.0
        iter_sbprec = 0.0
        iter_snowball = 0
        for it in range(eval_iter):
            if is_model2:
                batch_data = eval_dataset.next_multi_class(num_size=s_num_size, num_class=s_num_class)
                model(batch_data, s_num_size, s_num_class, threshold=threshold)
            else: 
                support_pos, support_neg, query, pos_class = eval_dataset.get_one_new_relation(self.train_data_loader, support_size, 10, query_size, query_class)
                model.forward(support_pos, support_neg, query, eval_distant_dataset, pos_class, threshold=threshold, threshold_for_snowball=threshold_for_snowball)
                model.forward_baseline(support_pos, support_neg, query, threshold=threshold)
                iter_bright += model._baseline_f1
                iter_bprec += model._baseline_prec
                iter_brecall += model._baseline_recall

            if hasattr(model, '_f1'):
                iter_right += model._f1
            else:
                iter_right += model._accuracy
            iter_prec += model._prec
            iter_recall += model._recall
            
            iter_sample += 1
            if hasattr(model, '_snowball'):
                snowball_cnt = model._snowball
                if model._snowball == 0:
                    snowball_prec = 0
                else:
                    snowball_prec = float(model._correct_snowball) / float(model._snowball)
                iter_sbprec += snowball_prec
            else:
                snowball_cnt = -1
                snowball_prec = -1
                iter_sbprec = 0
            iter_snowball += snowball_cnt
            sys.stdout.write('[EVAL tforsnow={0}] step: {1:4} | acc/f1: {2:1.4f}%, prec: {3:3.2f}%, recall: {4:3.2f}%, snowball: {5} | [baseline] acc/f1: {6:1.4f}%, prec: {7:3.2f}%, rec: {8:3.2f}%'.format(threshold_for_snowball, it + 1, iter_right / iter_sample, 100 * iter_prec / iter_sample, 100 * iter_recall / iter_sample, iter_snowball, iter_bright / iter_sample, 100 * iter_bprec / iter_sample, 100 * iter_brecall / iter_sample) +'\r')
            sys.stdout.flush()
        return iter_right / iter_sample

class PretrainFramework:

    def __init__(self, train_data_loader, val_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        if int(torch.__version__.split('.')[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train_encoder(self,
              model,
              model_name,
              batch_size=200,
              ckpt_dir='./checkpoint',
              learning_rate=1.,
              lr_step_size=10000,
              weight_decay=1e-5,
              train_iter=40000,
              val_iter=2000,
              val_step=1000,
              cuda=True,
              pretrain_model=None,
              optimizer=optim.SGD):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")
        model.train()
        
        # Init
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        opt = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_step_size)

        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        if cuda:
            model = model.cuda()

        # Training
        best_acc = 0
        not_best_count = 0 # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0

        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()
            batch_data = self.train_data_loader.next_batch(batch_size)
            model.forward_base(batch_data)
            loss = model.loss()
            right = model.accuracy()
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1

            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                print('')
                acc = self.eval_encoder(model, eval_iter=val_iter)
                print('')
                if acc > best_acc:
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_acc = acc
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval_encoder(self,
            model,
            eval_iter=2000,
            batch_size=500):
        '''
        model: a FewShotREModel instance
        eval_iter: num of iterations of evaluation
        return: Accuracy
        '''

        model.eval()
        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            batch_data = self.val_data_loader.next_batch(batch_size)
            model.forward_base(batch_data)
            right = model.accuracy()

            iter_right += self.item(right.data)
            iter_sample += 1

            sys.stdout.write('[EVAL] step: {0:4} | acc: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()
        model.train()
        return iter_right / iter_sample

    def train_siamese(self,
              model,
              model_name,
              batch_size=200,
              ckpt_dir='./checkpoint',
              learning_rate=1.,
              lr_step_size=10000,
              weight_decay=1e-5,
              train_iter=40000,
              val_iter=2000,
              val_step=1000,
              cuda=True,
              pretrain_model=None,
              optimizer=optim.SGD):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")
        model.train()
        
        # Init
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        opt = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_step_size)

        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        if cuda:
            model = model.cuda()

        # Training
        best_prec = 0
        not_best_count = 0 # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0

        s_num_size = 10
        s_num_class = 50

        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()

            batch_data = self.train_data_loader.next_multi_class(num_size=s_num_size, num_class=s_num_class)
            model(batch_data, s_num_size, s_num_class)

            loss = model._loss
            right = model._accuracy
            opt.zero_grad()
            loss.backward()
            opt.step()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, prec: {3:3.2f}%, recall: {4:3.2f}%'.format( \
                it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample, 100.0 * model._prec, 100.0 * model._recall) +'\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                print('')
                prec = self.eval_siamese(model, eval_iter=val_iter, threshold=0.5)
                print('')
                if prec > best_prec:
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_prec = prec
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval_siamese(self,
            model,
            s_num_size=10, s_num_class=10,
            eval_iter=2000,
            threshold=0.5):
        
        model.eval()
        iter_right = 0.0
        iter_prec = 0.0
        iter_recall = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            batch_data = self.val_data_loader.next_multi_class(num_size=s_num_size, num_class=s_num_class)
            model(batch_data, s_num_size, s_num_class, threshold=threshold)
            iter_right += model._accuracy
            iter_prec += model._prec
            iter_recall += model._recall
            iter_sample += 1

            sys.stdout.write('[EVAL] step: {0:4} | acc: {1:3.2f}%, prec: {2:3.2f}%, recall: {3:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample, 100 * iter_prec / iter_sample, 100 * iter_recall / iter_sample) + '\r')
            sys.stdout.flush()
        model.train()
        return iter_prec / iter_sample
