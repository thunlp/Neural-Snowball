import json
import os
import multiprocessing
import numpy as np
import random
import torch
from torch.autograd import Variable

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class FileDataLoader:
    def next_batch(self, B, N, K, Q):
        '''
        B: batch size.
        N: the number of relations for each batch
        K: the number of support instances for each relation
        Q: the number of query instances for each relation
        return: support_set, query_set, query_label
        '''
        raise NotImplementedError

class JSONFileDataLoaderBERT(FileDataLoader):
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data/bert'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        label_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_label.npy')
        entpair_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        rel2id_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2id.json')
        entpair2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(label_npy_file_name) or \
           not os.path.exists(entpair_npy_file_name) or \
           not os.path.exists(rel2scope_file_name) or \
           not os.path.exists(rel2id_file_name) or \
           not os.path.exists(entpair2scope_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.data_label = np.load(label_npy_file_name)
        self.data_entpair = np.load(entpair_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.rel2id = json.load(open(rel2id_file_name))
        self.entpair2scope = json.load(open(entpair2scope_file_name))
        self.instance_tot = self.data_word.shape[0]
        self.rel_tot = len(self.rel2id)
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, vocab, max_length=40, case_sensitive=False, reprocess=False, cuda=True, distant=False, rel2id=None, shuffle=True):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                            "token": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177": 
                    [
                        ...
                    ]
                ...
            }
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        cuda: Use cuda or not, default as True.
        '''
        self.file_name = file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.cuda = cuda
        self.shuffle = shuffle

        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            print("Finish loading")
            
            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.data_label = np.zeros((self.instance_tot), dtype=np.int32)
            self.data_entpair =[]
            self.rel2scope = {} # left close right open
            self.entpair2scope = {}
            if rel2id is not None:
                self.rel2id = rel2id
                self.rel_tot = len(self.rel2id)
            else:
                self.rel2id = {}
                self.rel_tot = 0
            i = 0

            tokenizer = BertTokenizer.from_pretrained(vocab)

            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                if relation not in self.rel2id:
                    self.rel2id[relation] = self.rel_tot
                    self.rel_tot += 1
                for ins in self.ori_data[relation]:
                    if distant:
                        head = ins['h']['name']
                        tail = ins['t']['name']
                        pos1 = ins['h']['pos'][0][0]
                        pos2 = ins['t']['pos'][0][0]
                        pos1_end = ins['h']['pos'][0][-1]
                        pos2_end = ins['t']['pos'][0][-1]
                    else:
                        head = ins['h'][0]
                        tail = ins['t'][0]
                        pos1 = ins['h'][2][0][0]
                        pos2 = ins['t'][2][0][0]
                        pos1_end = ins['h'][2][0][-1]
                        pos2_end = ins['t'][2][0][-1]
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]         
                    entpair = head + '#' + tail
                    self.data_entpair.append(entpair)

                    # tokenize
                    # # head entity # @ tail entity @
                    if pos1 < pos2:
                        new_words = ['[CLS]'] + words[:pos1] + ['#'] + words[pos1:pos1_end+1] + ['#'] + words[pos1_end+1:pos2] \
                                + ['@'] + words[pos2:pos2_end+1] + ['@'] + words[pos2_end+1:]
                    else:
                        new_words = ['[CLS]'] + words[:pos2] + ['@'] + words[pos2:pos2_end+1] + ['@'] + words[pos2_end+1:pos1] \
                                + ['#'] + words[pos1:pos1_end+1] + ['#'] + words[pos1_end+1:]
                    sentence = ' '.join(new_words)
                    tmp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
                    if len(tmp) < max_length:
                        tmp += [0] * (max_length - len(tmp))
                    else:
                        tmp = tmp[:max_length]
                    cur_ref_data_word = np.array(tmp)
                    
                    self.data_length[i] = len(words)
                    self.data_label[i] = self.rel2id[relation]
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if not entpair in self.entpair2scope:
                        self.entpair2scope[entpair] = [i]
                    else:
                        self.entpair2scope[entpair].append(i)
                    i += 1
                self.rel2scope[relation][1] = i 

            print("Finish pre-processing")     

            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            processed_data_dir = '_processed_data/bert'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            self.data_entpair = np.array(self.data_entpair)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            np.save(os.path.join(processed_data_dir, name_prefix + '_label.npy'), self.data_label)
            np.save(os.path.join(processed_data_dir, name_prefix + '_entpair.npy'), self.data_entpair)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            json.dump(self.rel2id, open(os.path.join(processed_data_dir, name_prefix + '_rel2id.json'), 'w'))
            json.dump(self.entpair2scope, open(os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json'), 'w'))
            print("Finish storing")
        
        self.id2rel = {}
        for rel in self.rel2id:
            self.id2rel[self.rel2id[rel]] = rel
        self.index = list(range(self.instance_tot))
        if self.shuffle:
            random.shuffle(self.index)
        self.current = 0

    def next_batch_one_epoch(self, batch_size):
        if self.current >= len(self.index):
            if self.shuffle:
                random.shuffle(self.index)
            self.current = 0
            return None
        batch = {'word': []}
        if self.current + batch_size > len(self.index):
            batch_size = len(self.index) - self.current
        current_index = self.index[self.current:self.current+batch_size]
        self.current += batch_size

        batch['word'] = Variable(torch.from_numpy(self.data_word[current_index]).long()) 
        batch['label']= Variable(torch.from_numpy(self.data_label[current_index]).long())

        # To cuda
        if self.cuda:
            for key in batch:
                batch[key] = batch[key].cuda()

        return batch

    def next_batch(self, batch_size):
        batch = {'word': []}
        if self.current + batch_size > len(self.index):
            self.index = list(range(self.instance_tot))
            if self.shuffle:
                random.shuffle(self.index)
            self.current = 0
        current_index = self.index[self.current:self.current+batch_size]
        self.current += batch_size

        batch['word'] = Variable(torch.from_numpy(self.data_word[current_index]).long()) 
        batch['label']= Variable(torch.from_numpy(self.data_label[current_index]).long())

        # To cuda
        if self.cuda:
            for key in batch:
                batch[key] = batch[key].cuda()

        return batch

    def next_support(self, support_size):
        support = {'word': []}
        for i in range(self.rel_tot):
            scope = self.rel2scope[self.id2rel[i]]
            indices = np.random.choice(list(range(scope[0], scope[1])), support_size, False)
            support['word'].append(self.data_word[indices])

        support['word'] = np.concatenate(support['word'], 0)

        support['word'] = Variable(torch.from_numpy(support['word']).long()) 

        # To cuda
        if self.cuda:
            for key in support:
                support[key] = support[key].cuda()

        return support


    def next_multi_class(self, num_size, num_class):
        '''
        num_size: The num of instances for ONE class. The total size is num_size * num_classes.
        num_class: The num of classes (include the positive class).
        '''
        target_classes = random.sample(self.rel2scope.keys(), num_class)
        batch = {'word': []}

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), num_size, False)
            batch['word'].append(self.data_word[indices])

        batch['word'] = np.concatenate(batch['word'], 0)

        batch['word'] = Variable(torch.from_numpy(batch['word']).long()) 

        # To cuda
        if self.cuda:
            for key in batch:
                batch[key] = batch[key].cuda()

        return batch

    def next_new_relation(self, train_data_loader, support_size, query_size, unlabelled_size, query_class, negative_rate=5, use_train_neg=False):
        '''
        support_size: The num of instances for positive / negative. The total support size is support_size * 2.
        query_size: The num of instances for ONE class in query set. The total query size is query_size * query_class.
        query_class: The num of classes in query (include the positive class).
        '''
        target_classes = random.sample(self.rel2scope.keys(), query_class) # 0 class is the new relation 
        support_set = {'word': []}
        query_set = {'word': [], 'label': []}
        unlabelled_set = {'word': []}

        # New relation
        scope = self.rel2scope[target_classes[0]]
        indices = np.random.choice(list(range(scope[0], scope[1])), support_size + query_size + unlabelled_size, False)
        support_word, query_word, unlabelled_word, _ = np.split(self.data_word[indices], [support_size, support_size + query_size, support_size + query_size + unlabelled_size])

        negative_support = train_data_loader.next_batch(support_size * negative_rate)
        support_set['word'] = np.concatenate((support_word, negative_support['word']), 0)
        support_set['label'] = np.concatenate((np.ones((support_size), dtype=np.int32), np.zeros((negative_rate * support_size), dtype=np.int32)), 0)

        query_set['word'].append(query_word)
        query_set['label'] += [1] * query_size

        unlabelled_set['word'].append(unlabelled_word)

        # Other query classes (negative)
        if use_train_neg:
            neg_loader = train_data_loader
            target_classes = random.sample(neg_loader.rel2scope.keys(), query_class) # discard 0 class 
        else:
            neg_loader = self

        for i, class_name in enumerate(target_classes[1:]):
            scope = neg_loader.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), query_size + unlabelled_size, False)
            query_word, unlabelled_word, _ = np.split(neg_loader.data_word[indices], [query_size, query_size + unlabelled_size])  

            query_set['word'].append(query_word)
            query_set['label'] += [0] * query_size

            unlabelled_set['word'].append(unlabelled_word)

        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['label'] = np.array(query_set['label'])

        unlabelled_set['word'] = np.concatenate(unlabelled_set['word'], 0)

        support_set['word'] = Variable(torch.from_numpy(support_set['word']).long()) 
        support_set['label'] = Variable(torch.from_numpy(support_set['label']).long())

        query_set['word'] = Variable(torch.from_numpy(query_set['word']).long()) 
        query_set['label'] = Variable(torch.from_numpy(query_set['label']).long())

        unlabelled_set['word'] = Variable(torch.from_numpy(unlabelled_set['word']).long()) 
 
        # To cuda
        if self.cuda:
            for key in support_set:
                support_set[key] = support_set[key].cuda()
            for key in query_set:
                query_set[key] = query_set[key].cuda()
            for key in unlabelled_set:
                unlabelled_set[key] = unlabelled_set[key].cuda()

        return support_set, query_set, unlabelled_set

    def get_same_entpair_ins(self, entpair):
        '''
        return instances with the same entpair
        entpair: a string with the format '$head_entity#$tail_entity'
        '''
        if not entpair in self.entpair2scope:
            return None
        scope = self.entpair2scope[entpair]
        batch = {}
        batch['word'] = Variable(torch.from_numpy(self.data_word[scope]).long()) 
        batch['label']= Variable(torch.from_numpy(self.data_label[scope]).long())
        batch['id']   = scope
        batch['entpair'] = entpair * len(scope)

        # To cuda
        if self.cuda:
            for key in ['word']:
                batch[key] = batch[key].cuda()

        return batch

    def get_random_candidate(self, pos_class, num_class, num_ins_per_class):
        '''
        random pick some instances for snowball phase 2 with total number num_class (1 pos + num_class-1 neg) * num_ins_per_class
        pos_class: positive relation (name)
        num_class: total number of classes, including the positive and negative relations
        num_ins_per_class: the number of instances of each relation
        return: a dataset
        '''
        
        target_classes = random.sample(self.rel2scope.keys(), num_class) 
        if not pos_class in target_classes:
            target_classes = target_classes[:-1] + [pos_class]
        candidate = {'word': [], 'id': [], 'entpair': []}

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), min(num_ins_per_class, scope[1] - scope[0]), False)
            candidate['word'].append(self.data_word[indices])
            candidate['id'] += list(indices)
            candidate['entpair'] += list(self.data_entpair[indices])

        candidate['word'] = np.concatenate(candidate['word'], 0)

        candidate['word'] = Variable(torch.from_numpy(candidate['word']).long()) 

        # To cuda
        if self.cuda:
            for key in ['word']:
                candidate[key] = candidate[key].cuda()

        return candidate
   
    def get_one_new_relation(self, train_data_loader, support_pos_size, support_neg_rate, query_size, query_class, use_train_neg=False):
        '''
        get data for one new relation
        train_data_loader: training data loader
        support_pos_size: num of ins of ONE new relation in support set
        support_neg_rate: num of neg ins in support is support_neg_rate times as pos
        query_size: num of ins for EACH class in query set
        query_class: num of classes in query set
        return: support_pos, support_neg, query, name_of_pos_class
        '''
        target_classes = random.sample(self.rel2scope.keys(), query_class) # 0 class is the new relation 
        support_pos = {'word': [], 'id': [], 'entpair': []}
        support_neg = {'word': [], 'id': [], 'entpair': []}
        query = {'word': [], 'label': []}

        # New relation
        scope = self.rel2scope[target_classes[0]]
        indices = np.random.choice(list(range(scope[0], scope[1])), support_pos_size + query_size, False)
        support_word, query_word, _ = np.split(self.data_word[indices], [support_pos_size, support_pos_size + query_size])
        support_id = list(indices[:support_pos_size])
        support_entpair = list(self.data_entpair[indices[:support_pos_size]])

        support_neg = train_data_loader.next_batch(support_pos_size * support_neg_rate)
        support_neg['label'] = np.zeros((support_neg_rate * support_pos_size), dtype=np.int32)

        support_pos['word'] = support_word
        support_pos['label'] = np.ones((support_pos_size), dtype=np.int32)
        support_pos['id'] = support_id
        support_pos['entpair'] = support_entpair

        query['word'].append(query_word)
        query['label'] += [1] * query_size

        # Other query classes (negative)
        if use_train_neg:
            neg_loader = train_data_loader
            target_classes = random.sample(neg_loader.rel2scope.keys(), query_class) # discard 0 class 
        else:
            neg_loader = self

        for i, class_name in enumerate(target_classes[1:]):
            scope = neg_loader.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), query_size, False)
            query['word'].append(neg_loader.data_word[indices])  
            query['label'] += [0] * query_size

        query['word'] = np.concatenate(query['word'], 0)
        query['label'] = np.array(query['label'])

        support_pos['word'] = Variable(torch.from_numpy(support_pos['word']).long()) 
        support_pos['label'] = Variable(torch.from_numpy(support_pos['label']).long())
        support_neg['label'] = Variable(torch.from_numpy(support_neg['label']).long())

        query['word'] = Variable(torch.from_numpy(query['word']).long()) 
        query['label'] = Variable(torch.from_numpy(query['label']).long())

        # To cuda
        if self.cuda:
            for key in ['word', 'label']:
                support_pos[key] = support_pos[key].cuda()
            for key in ['word', 'label']:
                query[key] = query[key].cuda()
            support_neg[key] = support_neg[key].cuda()

        return support_pos, support_neg, query, target_classes[0]

    def get_all(self, rel):
        batch = {'word': []}
        current_index = list(range(self.rel2scope[rel][0], self.rel2scope[rel][1]))
        batch['word'] = Variable(torch.from_numpy(self.data_word[current_index]).long()) 
        batch['label']= Variable(torch.from_numpy(self.data_label[current_index]).long())

        # To cuda
        if self.cuda:
            for key in batch:
                batch[key] = batch[key].cuda()

        return batch

    def get_10shot_query(self, target_class, iid):
        '''
        get data for one new relation
        train_data_loader: training data loader
        support_pos_size: num of ins of ONE new relation in support set
        support_neg_rate: num of neg ins in support is support_neg_rate times as pos
        query_size: num of ins for EACH class in query set
        query_class: num of classes in query set
        return: support_pos, support_neg, query, name_of_pos_class
        '''
        query = {'word': [], 'label': []}

        scope = self.rel2scope[target_class]
        indices = list(range(scope[0], scope[1]))
        query['word'] = self.data_word[indices]
        query['label'] = [iid] * query['word'].shape[0]
        query['label'] = np.array(query['label'])

        query['word'] = Variable(torch.from_numpy(query['word']).long()) 
        query['label'] = Variable(torch.from_numpy(query['label']).long())

        # To cuda
        if self.cuda:
            for key in ['word', 'label']:
                query[key] = query[key].cuda()

        return query

    def get_all_new_relation10shot(self, target_class, query_data_loader, query_size):
        '''
        get data for one new relation
        train_data_loader: training data loader
        support_pos_size: num of ins of ONE new relation in support set
        support_neg_rate: num of neg ins in support is support_neg_rate times as pos
        query_size: num of ins for EACH class in query set
        query_class: num of classes in query set
        return: support_pos, support_neg, query, name_of_pos_class
        '''
        support_pos = {'word': [], 'id': [], 'entpair': []}
        support_neg = {'word': [],  'id': [], 'entpair': []}

        # New relation
        scope = self.rel2scope[target_class]
        indices = list(range(scope[0], scope[1]))
        support_pos['word'] = self.data_word[indices]
        support_pos['label'] = np.ones((support_pos['word'].shape[0]), dtype=np.int32)
        support_pos['id'] = indices 
        support_pos['entpair'] = list(self.data_entpair[indices])

        for rel in self.rel2scope:
            if rel != target_class:
                scope = self.rel2scope[rel]
                indices = list(range(scope[0], scope[1]))
                support_neg['word'].append(self.data_word[indices])

        support_neg['word'] = np.concatenate(support_neg['word'], 0) 
        support_neg['label'] = np.zeros((support_neg['word'].shape[0]), dtype=np.int32)
       
        query = {'word': [], 'label': []}

        for class_name in query_data_loader.rel2scope:
            scope = query_data_loader.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), query_size, False)
            query['word'].append(query_data_loader.data_word[indices])  
            if class_name == target_class:
                query['label'] += [1] * query_size
            else:
                query['label'] += [0] * query_size

        query['word'] = np.concatenate(query['word'], 0)
        query['label'] = np.array(query['label'])

        query['word'] = Variable(torch.from_numpy(query['word']).long()) 
        query['label'] = Variable(torch.from_numpy(query['label']).long())

        support_pos['word'] = Variable(torch.from_numpy(support_pos['word']).long()) 
        support_neg['word'] = Variable(torch.from_numpy(support_neg['word']).long()) 

        support_pos['label'] = Variable(torch.from_numpy(support_pos['label']).long())
        support_neg['label'] = Variable(torch.from_numpy(support_neg['label']).long())

        # To cuda
        if self.cuda:
            for key in ['word', 'label']:
                support_pos[key] = support_pos[key].cuda()
                support_neg[key] = support_neg[key].cuda()
                query[key] = query[key].cuda()

        return support_pos, support_neg, query

