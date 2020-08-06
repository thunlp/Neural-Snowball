import os
import sklearn.metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import argparse

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

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
        self.parser = argparse.ArgumentParser()
        self.NA_label = None
    
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
        if self.NA_label is not None:
            pred = pred.view(-1).cpu().detach().numpy()
            label = label.view(-1).cpu().detach().numpy()
            return float(np.logical_and(label != self.NA_label, label == pred).sum()) / float((label != self.NA_label).sum() + 1)
        else:
            return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor)).item()
    
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

        s_num_class = 8
        s_num_size = batch_size // s_num_class

        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()
            batch_data = self.train_data_loader.next_batch(batch_size)
            model.forward_base(batch_data)
            loss = model.loss()
            right = model.accuracy()
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            iter_loss += loss
            iter_right += right
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

                iter_loss2 += loss2
                iter_right2 += right2
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

    def eval_10shot(self,
            model,
            support_size=10, query_size=50, unlabelled_size=50, query_class=5,
            s_num_size=10, s_num_class=10,
            eval_iter=2000,
            ckpt=None,
            is_model2=False,
            threshold=0.5,
            threshold_for_snowball=0.5):
        '''
        test by val_support.json (10shot) / val_query.json
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
        snowball_metric = [np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32) ]
        
        Ws = []
        bs = []

        for iid, rel in enumerate(self.val_support_data_loader.rel2scope):
            support_pos, support_neg, query = self.val_support_data_loader.get_all_new_relation10shot(rel, self.val_query_data_loader, query_size)
            model.forward_baseline(support_pos, support_neg, query, threshold=threshold)
            model.forward(support_pos, support_neg, query, eval_distant_dataset, rel, threshold=threshold, threshold_for_snowball=threshold_for_snowball)
            Ws.append(model.new_W)
            bs.append(model.new_bias)

            iter_bright += model._baseline_f1
            iter_bprec += model._baseline_prec
            iter_brecall += model._baseline_recall

            iter_right += model._f1
            iter_prec += model._prec
            iter_recall += model._recall
            
            iter_sample += 1
            sys.stdout.write('[EVAL] step: {0:4} | f1: {1:1.4f}, prec: {2:3.2f}%, recall: {3:3.2f}% | [baseline] f1: {4:1.4f}, prec: {5:3.2f}%, rec: {6:3.2f}%'.format(iid, iter_right / iter_sample, 100 * iter_prec / iter_sample, 100 * iter_recall / iter_sample, iter_bright / iter_sample, 100 * iter_bprec / iter_sample, 100 * iter_brecall / iter_sample) +'\r')
            sys.stdout.flush()
            
            print("")
            print("[SNOWBALL ITER RESULT:]")
            for i in range(len(model._metric)):
                snowball_metric[i] += model._metric[i]
                print("iter {} : {}".format(i, snowball_metric[i] / iter_sample))

        model.init_10shot(Ws, bs)
        
        acc = 0.0
        for iid, rel in enumerate(self.val_support_data_loader.rel2scope):
            query = self.val_query_data_loader.get_10shot_query(rel, iid)
            acc += model.eval_10shot(query).item()
            print("current acc: {}".format(acc / (iid + 1)))
        acc = acc / self.val_support_data_loader.rel_tot
        print("FINAL ACC: {}".format(acc))
        print("")

        return iter_right / iter_sample


    def eval_selected(self,
            model,
            support_size=10, query_size=600, unlabelled_size=50, query_class=16,
            ckpt=None,
            is_model2=False,
            threshold=0.5):
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
        snowball_metric = [np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32) ]
        # for rel in ['P2094']:
        for rel in self.val_data_loader.rel2scope:
            support_pos, support_neg, query, pos_class = eval_dataset.get_selected(self.train_data_loader, support_size, 10, query_size, query_class, main_class=rel, use_train_neg=True, neg_train_loader=self.neg_train_loader)

            model.forward_baseline(support_pos, support_neg, query, threshold=threshold)
            model.forward(support_pos, support_neg, query, eval_distant_dataset, pos_class, threshold=threshold)

            iter_bright += model._baseline_f1
            iter_bprec += model._baseline_prec
            iter_brecall += model._baseline_recall

            iter_right += model._f1
            iter_prec += model._prec
            iter_recall += model._recall
            
            iter_sample += 1
            # sys.stdout.write('[EVAL] step: {0:4} | f1: {1:1.4f}, prec: {2:3.2f}%, recall: {3:3.2f}% | [baseline] f1: {4:1.4f}, prec: {5:3.2f}%, rec: {6:3.2f}%'.format(it + 1, iter_right / iter_sample, 100 * iter_prec / iter_sample, 100 * iter_recall / iter_sample, iter_bright / iter_sample, 100 * iter_bprec / iter_sample, 100 * iter_brecall / iter_sample) +'\r')
            # sys.stdout.flush()

            print("------- RESULT FOR {} -------".format(rel))
            print("BASELINE: f1: {}, precision: {}, recall: {}".format(model._baseline_f1, model._baseline_prec, model._baseline_recall))
            print("SNOWBALL: f1: {}, precision: {}, recall: {}".format(model._f1, model._prec, model._recall))
            
            print("")
            print("[SNOWBALL ITER RESULT:]")
            for i in range(len(model._metric)):
                snowball_metric[i] += model._metric[i]
                print("iter {} : {}".format(i, snowball_metric[i] / iter_sample))

        print("------- OVERALL RESULT -------")
        print("BASELINE: f1: {}, precision: {}, recall: {}".format(iter_bright / iter_sample, iter_bprec / iter_sample, iter_brecall / iter_sample))
        print("SNOWBALL: f1: {}, precision: {}, recall: {}".format(iter_right / iter_sample, iter_prec / iter_sample, iter_recall / iter_sample))

        res = "[SNOWBALL ITER RESULT:]\n"
        for i in range(len(model._metric)):
            res += "iter {} : {}\n".format(i, snowball_metric[i] / iter_sample)
        res += "BASELINE: f1: {}, precision: {}, recall: {}".format(iter_bright / iter_sample, iter_bprec / iter_sample, iter_brecall / iter_sample) + "| SNOWBALL: f1: {}, precision: {}, recall: {}\n".format(iter_right / iter_sample, iter_prec / iter_sample, iter_recall / iter_sample)
        res += '--------\n\n'
        return res

    def eval_baseline(self,
            model,
            support_size=10, query_size=50, unlabelled_size=50,
            s_num_size=10, s_num_class=10,
            eval_iter=2000,
            ckpt=None,
            is_model2=False,
            threshold=0.5):
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
        snowball_metric = [np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32) ]
        for it in range(eval_iter):
            support_pos, query, pos_class = eval_dataset.sample_for_eval(self.train_data_loader, support_size, query_size)
            model.forward_baseline(support_pos, query, threshold=threshold)

            # support_pos, support_neg, query, pos_class = eval_dataset.get_one_new_relation(self.train_data_loader, support_size, 10, query_size, query_class, use_train_neg=True, neg_train_loader=self.neg_train_loader)
            # model.forward_baseline(support_pos, support_neg, query, threshold=threshold)
            # model.forward(support_pos, support_neg, query, eval_distant_dataset, pos_class, threshold=threshold)

            iter_bright += model._baseline_f1
            iter_bprec += model._baseline_prec
            iter_brecall += model._baseline_recall

            iter_sample += 1
            sys.stdout.write('[EVAL] step: {0:4} | [baseline] f1: {1:1.4f}, prec: {2:3.2f}%, rec: {3:3.2f}%'.format(it + 1, iter_bright / iter_sample, 100 * iter_bprec / iter_sample, 100 * iter_brecall / iter_sample) +'\r')
            sys.stdout.flush()
        
        print('')
        return '[EVAL] step: {0:4} | [baseline] f1: {1:1.4f}, prec: {2:3.2f}%, rec: {3:3.2f}%'.format(it + 1, iter_bright / iter_sample, 100 * iter_bprec / iter_sample, 100 * iter_brecall / iter_sample)
       #  return iter_right / iter_sample

    def eval(self,
            model,
            support_size=10, query_size=50, unlabelled_size=50, query_class=5,
            s_num_size=10, s_num_class=10,
            eval_iter=100,
            ckpt=None,
            is_model2=False,
            threshold=0.5,
            query_train=False, query_val=True):
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
        model.eval()
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
        snowball_metric = [np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32) ]
        for it in range(eval_iter):
            # support_pos, support_neg, query, pos_class = eval_dataset.get_one_new_relation(self.train_data_loader, support_size, 10, query_size, query_class, use_train_neg=True, neg_train_loader=self.neg_train_loader)
            support_pos, query, pos_class = eval_dataset.sample_for_eval(self.train_data_loader, support_size, query_size, query_train=query_train, query_val=query_val)
            model.forward_baseline(support_pos, query, threshold=threshold)
            model.forward(support_pos, query, eval_distant_dataset, pos_class, threshold=threshold)

            iter_bright += model._baseline_f1
            iter_bprec += model._baseline_prec
            iter_brecall += model._baseline_recall

            iter_right += model._f1
            iter_prec += model._prec
            iter_recall += model._recall
            
            iter_sample += 1
            sys.stdout.write('[EVAL] step: {0:4} | f1: {1:1.4f}, prec: {2:3.2f}%, recall: {3:3.2f}% | [baseline] f1: {4:1.4f}, prec: {5:3.2f}%, rec: {6:3.2f}%'.format(it + 1, iter_right / iter_sample, 100 * iter_prec / iter_sample, 100 * iter_recall / iter_sample, iter_bright / iter_sample, 100 * iter_bprec / iter_sample, 100 * iter_brecall / iter_sample) +'\r')
            sys.stdout.flush()
            
            if model.args.eval:
                print("")
                print("[SNOWBALL ITER RESULT:]")
                for i in range(len(model._metric)):
                    snowball_metric[i] += model._metric[i]
                    print("iter {} : {}".format(i, snowball_metric[i] / iter_sample))

        res = "{} {} {} {} {} {}".format(iter_bright / iter_sample, iter_bprec / iter_sample, iter_brecall / iter_sample, iter_right / iter_sample, iter_prec / iter_sample, iter_recall / iter_sample)
        return res

    def eval_fewshot(self,
            model,
            support_size=5, query_size=50, unlabelled_size=50, query_class=5,
            batch_size=100,
            eval_iter=2000,
            ckpt=None,
            is_model2=False,
            threshold=0.5):
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
        # snowball_metric = [np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32) ]
        for it in range(eval_iter):
            support, query, label = self.next_fewshot_batch(B=batch_size, N=query_class, K=support_size, Q=query_size)
            model.forward_few_shot_baseline(support, query, label, B=batch_size, N=query_class, K=support_size)
            model.forward_few_shot(support, query, label, B=batch_size, N=query_class, K=support_size)

            iter_bright += model._baseline_acc
            iter_right += model._acc
            
            iter_sample += 1
            sys.stdout.write('[EVAL] step: {0:4} | acc: {1:1.4f} | [baseline] acc: {2:1.4f}%'.format(it + 1, iter_right / iter_sample * 100, iter_bright / iter_sample * 100) +'\r')
            sys.stdout.flush()
            
            # print("")
            # print("[SNOWBALL ITER RESULT:]")
            # for i in range(len(model._metric)):
            #     snowball_metric[i] += model._metric[i]
            #     print("iter {} : {}".format(i, snowball_metric[i] / iter_sample))
        return iter_right / iter_sample

class SuperviseFramework:

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
  
    
    def train_encoder_epoch(self,
              model,
              model_name,
              optimizer,
              batch_size=200,
              ckpt_dir='./checkpoint',
              learning_rate=1.,
              weight_decay=1e-5,
              train_epoch=30,
              cuda=True,
              pretrain_model=None,
              support=False,
              support_size=10,
              warmup=True,
              warmup_step=300,
              grad_iter=1):
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
        
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            # start_iter = checkpoint['iter'] + 1
            start_iter = 0
        else:
            start_iter = 0

        if cuda:
            model = model.cuda()

        # Training
        best_acc = 0
        global_step = 0
        
        for epoch in range(train_epoch):
            epoch_step = 0
            iter_loss = 0.0
            iter_right = 0.0
            iter_sample = 0.0
            while True:
                global_step += 1
                epoch_step += 1
                batch_data = self.train_data_loader.next_batch_one_epoch(batch_size)
                if batch_data is None:
                    break
      
                if support:
                    support_data = self.train_data_loader.next_support(support_size)
                    model.forward_base(batch_data, support_data)
                else:
                    model.forward_base(batch_data)
                loss = model.loss() / grad_iter
                right = model.accuracy()
                loss.backward()
                
                # warmup
                cur_lr = learning_rate
                if warmup:
                    cur_lr *= warmup_linear(global_step, warmup_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr

                if global_step % grad_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                iter_loss += loss
                iter_right += right
                iter_sample += 1

                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(epoch_step, iter_loss / iter_sample, 100 * iter_right / iter_sample) +'\r')
                sys.stdout.flush()

            print('')
            acc = self.eval_encoder_one_epoch(model, support=support, batch_size=batch_size)
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

    def train_encoder(self,
              model,
              model_name,
              batch_size=200,
              ckpt_dir='./checkpoint',
              learning_rate=1.,
              lr_step_size=30000,
              weight_decay=1e-5,
              train_iter=60000,
              val_iter=2000,
              val_step=2000,
              cuda=True,
              pretrain_model=None,
              support=False,
              support_size=10,
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
            # start_iter = checkpoint['iter'] + 1
            start_iter = 0
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
      
            if support:
                support_data = self.train_data_loader.next_support(support_size)
                model.forward_base(batch_data, support_data)
            else:
                model.forward_base(batch_data)
            loss = model.loss()
            right = model.accuracy()
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            iter_loss += loss
            iter_right += right
            iter_sample += 1

            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                print('')
                acc = self.eval_encoder(model, support=support, eval_iter=val_iter)
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

    def eval_encoder_one_epoch(self,
            model,
            batch_size=200,
            support_size=10,
            support=False):
        '''
        model: a FewShotREModel instance
        eval_iter: num of iterations of evaluation
        return: Accuracy
        '''

        model.eval()
        iter_right = 0.0
        iter_sample = 0.0
        it = 0
        pred = []
        label = []
        while True:
            it += 1
            batch_data = self.val_data_loader.next_batch_one_epoch(batch_size)
            if batch_data is None:
                break
            if support:
                support_data = self.val_data_loader.next_support(support_size)
                model.forward_base(batch_data, support_data)
            else:
                model.forward_base(batch_data)
            right = model.accuracy()

            iter_right += right
            iter_sample += 1

            sys.stdout.write('[EVAL] step: {0:4} | acc: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            pred.append(model._pred.cpu().detach().numpy())
            label.append(batch_data['label'].cpu().detach().numpy())
            
        model.train()
        pred = np.concatenate(pred)
        label = np.concatenate(label)
        pred = label_binarize(pred, classes=list(range(0, 13)) + list(range(14, self.val_data_loader.rel_tot)))
        label = label_binarize(label, classes=list(range(0, 13)) + list(range(14, self.val_data_loader.rel_tot)))

        micro_precision = sklearn.metrics.average_precision_score(pred, label, average='micro')
        micro_recall = sklearn.metrics.recall_score(pred, label, average='micro')
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        print('')
        print('micro precision: {}, micro recall: {}, micro f1: {}'.format(micro_precision, micro_recall, micro_f1))
        print('')

        return iter_right / iter_sample

    def eval_encoder(self,
            model,
            eval_iter=2000,
            batch_size=200,
            support_size=10,
            support=False):
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
            if support:
                support_data = self.val_data_loader.next_support(support_size)
                model.forward_base(batch_data, support_data)
            else:
                model.forward_base(batch_data)
            right = model.accuracy()

            iter_right += right
            iter_sample += 1

            sys.stdout.write('[EVAL] step: {0:4} | acc: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()
        model.train()
        return iter_right / iter_sample

    def train_siamese(self,
              model,
              model_name,
              optimizer,
              batch_size=200,
              ckpt_dir='./checkpoint',
              learning_rate=1.,
              train_iter=30000,
              val_iter=2000,
              val_step=2000,
              cuda=True,
              pretrain_model=None,
              warmup=True,
              warmup_step=300,
              s_num_class=8,
              grad_iter=1):
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
        
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        if cuda:
            model = model.cuda()

        # Training
        global_step = 0

        best_prec = 0
        not_best_count = 0 # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        iter_prec = 0.0
        iter_recall = 0.0

        s_num_size = batch_size // s_num_class

        for it in range(start_iter, start_iter + train_iter):
            global_step += 1

            batch_data = self.train_data_loader.next_multi_class(num_size=s_num_size, num_class=s_num_class)
            model(batch_data, s_num_size, s_num_class)

            loss = model._loss / grad_iter
            right = model._accuracy
            loss.backward()

            # warmup
            cur_lr = learning_rate
            if warmup:
                cur_lr *= warmup_linear(global_step, warmup_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
            
            if global_step % grad_iter == 0:
                optimizer.step()
                optimizer.zero_grad()

            iter_loss += loss
            iter_right += right
            iter_sample += 1
            iter_prec += model._prec
            iter_recall += model._recall
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, prec: {3:3.2f}%, recall: {4:3.2f}%'.format( \
                it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample, 100.0 * iter_prec / iter_sample, 100.0 * iter_recall / iter_sample) +'\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.
                iter_prec = 0.
                iter_recall = 0.

            if (it + 1) % val_step == 0:
                print('')
                prec = self.eval_siamese(model, eval_iter=val_iter, threshold=0.5, s_num_size=s_num_size, s_num_class=s_num_class)
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
