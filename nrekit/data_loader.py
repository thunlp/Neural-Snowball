import json
import os
import multiprocessing
import numpy as np
import random
import torch
from torch.autograd import Variable

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

class JSONFileDataLoader(FileDataLoader):
    def _load_preprocessed_file(self): 
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        self.uid = np.load('./data/' + name_prefix + '_uid.npy')
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        label_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_label.npy')
        entpair_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        rel2id_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2id.json')
        entpair2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(mask_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(label_npy_file_name) or \
           not os.path.exists(entpair_npy_file_name) or \
           not os.path.exists(rel2scope_file_name) or \
           not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name) or \
           not os.path.exists(rel2id_file_name) or \
           not os.path.exists(entpair2scope_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.data_label = np.load(label_npy_file_name)
        self.data_entpair = np.load(entpair_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        self.rel2id = json.load(open(rel2id_file_name))
        self.entpair2scope = json.load(open(entpair2scope_file_name))
        self.instance_tot = self.data_word.shape[0]
        self.rel_tot = len(self.rel2id)
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=False, cuda=True, distant=False, rel2id=None, shuffle=True):
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
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        cuda: Use cuda or not, default as True.
        '''
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.cuda = cuda
        self.shuffle = shuffle

        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")
            
            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

      
            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            UNK = self.word_vec_tot
            BLANK = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
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
                    else:
                        head = ins['h'][0]
                        tail = ins['t'][0]
                        pos1 = ins['h'][2][0][0]
                        pos2 = ins['t'][2][0][0]
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]         
                    entpair = head + '#' + tail
                    self.data_entpair.append(entpair)

                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]
                            else:
                                cur_ref_data_word[j] = UNK
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = BLANK
                    self.data_length[i] = len(words)
                    self.data_label[i] = self.rel2id[relation]
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                        if j >= self.data_length[i]:
                            self.data_mask[i][j] = 0
                        elif j <= pos_min:
                            self.data_mask[i][j] = 1
                        elif j <= pos_max:
                            self.data_mask[i][j] = 2
                        else:
                            self.data_mask[i][j] = 3
                    if not entpair in self.entpair2scope:
                        self.entpair2scope[entpair] = [i]
                    else:
                        self.entpair2scope[entpair].append(i)
                    i += 1
                self.rel2scope[relation][1] = i 

            print("Finish pre-processing")     

            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            self.data_entpair = np.array(self.data_entpair)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            np.save(os.path.join(processed_data_dir, name_prefix + '_label.npy'), self.data_label)
            np.save(os.path.join(processed_data_dir, name_prefix + '_entpair.npy'), self.data_entpair)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
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
        batch = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
        if self.current + batch_size > len(self.index):
            batch_size = len(self.index) - self.current
        current_index = self.index[self.current:self.current+batch_size]
        self.current += batch_size

        batch['word'] = Variable(torch.from_numpy(self.data_word[current_index]).long()) 
        batch['pos1'] = Variable(torch.from_numpy(self.data_pos1[current_index]).long())
        batch['pos2'] = Variable(torch.from_numpy(self.data_pos2[current_index]).long())
        batch['mask'] = Variable(torch.from_numpy(self.data_mask[current_index]).long())
        batch['label']= Variable(torch.from_numpy(self.data_label[current_index]).long())
        batch['id'] = Variable(torch.from_numpy(self.uid[current_index]).long())

        # To cuda
        if self.cuda:
            for key in batch:
                batch[key] = batch[key].cuda()

        return batch

    def next_batch(self, batch_size):
        batch = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
        if self.current + batch_size > len(self.index):
            self.index = list(range(self.instance_tot))
            if self.shuffle:
                random.shuffle(self.index)
            self.current = 0
        current_index = self.index[self.current:self.current+batch_size]
        self.current += batch_size

        batch['word'] = Variable(torch.from_numpy(self.data_word[current_index]).long()) 
        batch['pos1'] = Variable(torch.from_numpy(self.data_pos1[current_index]).long())
        batch['pos2'] = Variable(torch.from_numpy(self.data_pos2[current_index]).long())
        batch['mask'] = Variable(torch.from_numpy(self.data_mask[current_index]).long())
        batch['id'] = Variable(torch.from_numpy(self.uid[current_index]).long())
        batch['label']= Variable(torch.from_numpy(self.data_label[current_index]).long())

        # To cuda
        if self.cuda:
            for key in batch:
                batch[key] = batch[key].cuda()

        return batch

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
        batch['pos1'] = Variable(torch.from_numpy(self.data_pos1[scope]).long())
        batch['pos2'] = Variable(torch.from_numpy(self.data_pos2[scope]).long())
        batch['mask'] = Variable(torch.from_numpy(self.data_mask[scope]).long())
        batch['label']= Variable(torch.from_numpy(self.data_label[scope]).long())
        batch['id'] = Variable(torch.from_numpy(self.uid[scope]).long())
        batch['entpair'] = [entpair] * len(scope)

        # To cuda
        if self.cuda:
            for key in ['word', 'pos1', 'pos2', 'mask', 'id']:
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
        candidate = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), min(num_ins_per_class, scope[1] - scope[0]), False)
            candidate['word'].append(self.data_word[indices])
            candidate['pos1'].append(self.data_pos1[indices])
            candidate['pos2'].append(self.data_pos2[indices])
            candidate['mask'].append(self.data_mask[indices])
            candidate['id'].append(self.uid[indices])
            candidate['entpair'] += list(self.data_entpair[indices])

        candidate['word'] = np.concatenate(candidate['word'], 0)
        candidate['pos1'] = np.concatenate(candidate['pos1'], 0)
        candidate['pos2'] = np.concatenate(candidate['pos2'], 0)
        candidate['mask'] = np.concatenate(candidate['mask'], 0)
        candidate['id'] = np.concatenate(candidate['id'], 0)

        candidate['word'] = Variable(torch.from_numpy(candidate['word']).long()) 
        candidate['pos1'] = Variable(torch.from_numpy(candidate['pos1']).long())
        candidate['pos2'] = Variable(torch.from_numpy(candidate['pos2']).long())
        candidate['mask'] = Variable(torch.from_numpy(candidate['mask']).long())
        candidate['id'] = Variable(torch.from_numpy(candidate['id']).long())

        # To cuda
        if self.cuda:
            for key in ['word', 'pos1', 'pos2', 'mask', 'id']:
                candidate[key] = candidate[key].cuda()

        return candidate
   
    def get_one_new_relation(self, train_data_loader, support_pos_size, support_neg_rate, query_size, query_class, use_train_neg=False, neg_train_loader=None):
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
        support_pos = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
        support_neg = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'label': []}

        # New relation
        scope = self.rel2scope[target_classes[0]]
        indices = np.random.choice(list(range(scope[0], scope[1])), support_pos_size + query_size, False)
        support_word, query_word, _ = np.split(self.data_word[indices], [support_pos_size, support_pos_size + query_size])
        support_pos1, query_pos1, _ = np.split(self.data_pos1[indices], [support_pos_size, support_pos_size + query_size])
        support_pos2, query_pos2, _ = np.split(self.data_pos2[indices], [support_pos_size, support_pos_size + query_size])
        support_mask, query_mask, _ = np.split(self.data_mask[indices], [support_pos_size, support_pos_size + query_size])
        support_id, query_id, _ = np.split(self.uid[indices], [support_pos_size, support_pos_size + query_size])
        support_entpair = list(self.data_entpair[indices[:support_pos_size]])
        
        if neg_train_loader is None:
            neg_train_loader = train_data_loader
        support_neg = neg_train_loader.next_batch(support_pos_size * support_neg_rate)
        support_neg['label'] = np.zeros((support_neg_rate * support_pos_size), dtype=np.int32)

        support_pos['word'] = support_word
        support_pos['pos1'] = support_pos1
        support_pos['pos2'] = support_pos2
        support_pos['mask'] = support_mask
        support_pos['label'] = np.ones((support_pos_size), dtype=np.int32)
        support_pos['id'] = support_id
        support_pos['entpair'] = support_entpair

        query['word'].append(query_word)
        query['pos1'].append(query_pos1) 
        query['pos2'].append(query_pos2)
        query['mask'].append(query_mask)
        query['id'].append(query_id)
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
            query['pos1'].append(neg_loader.data_pos1[indices])    
            query['pos2'].append(neg_loader.data_pos2[indices])    
            query['mask'].append(neg_loader.data_mask[indices])
            query['id'].append(neg_loader.uid[indices])
            query['label'] += [0] * query_size

        query['word'] = np.concatenate(query['word'], 0)
        query['pos1'] = np.concatenate(query['pos1'], 0)
        query['pos2'] = np.concatenate(query['pos2'], 0)
        query['mask'] = np.concatenate(query['mask'], 0)
        query['id'] = np.concatenate(query['id'], 0)
        query['label'] = np.array(query['label'])

        support_pos['word'] = Variable(torch.from_numpy(support_pos['word']).long()) 
        support_pos['pos1'] = Variable(torch.from_numpy(support_pos['pos1']).long())
        support_pos['pos2'] = Variable(torch.from_numpy(support_pos['pos2']).long())
        support_pos['mask'] = Variable(torch.from_numpy(support_pos['mask']).long())
        support_pos['id'] = Variable(torch.from_numpy(support_pos['id']).long())
        support_pos['label'] = Variable(torch.from_numpy(support_pos['label']).long())

        support_neg['label'] = Variable(torch.from_numpy(support_neg['label']).long())

        query['word'] = Variable(torch.from_numpy(query['word']).long()) 
        query['pos1'] = Variable(torch.from_numpy(query['pos1']).long())
        query['pos2'] = Variable(torch.from_numpy(query['pos2']).long())
        query['mask'] = Variable(torch.from_numpy(query['mask']).long())
        query['id'] = Variable(torch.from_numpy(query['id']).long())
        query['label'] = Variable(torch.from_numpy(query['label']).long())

        # To cuda
        if self.cuda:
            for key in ['word', 'pos1', 'pos2', 'mask', 'label', 'id']:
                support_pos[key] = support_pos[key].cuda()
                support_neg[key] = support_neg[key].cuda()
                query[key] = query[key].cuda()
            support_neg[key] = support_neg[key].cuda()

        return support_pos, support_neg, query, target_classes[0]

    def get_selected(self, train_data_loader, support_pos_size, support_neg_rate, query_size, query_class, main_class, use_train_neg=False, neg_train_loader=None):
        '''
        get data for one new relation
        train_data_loader: training data loader
        support_pos_size: num of ins of ONE new relation in support set
        support_neg_rate: num of neg ins in support is support_neg_rate times as pos
        query_size: num of ins for EACH class in query set
        query_class: num of classes in query set
        return: support_pos, support_neg, query, name_of_pos_class
        '''
        target_classes = self.rel2scope.keys()
        support_pos = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
        support_neg = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': [], 'label': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'label': [], 'entpair': [], 'id': []}

        # New relation
        scope = self.rel2scope[main_class]
        indices = list(range(scope[0], scope[0] + support_pos_size + query_size))
        support_word, query_word, _ = np.split(self.data_word[indices], [support_pos_size, support_pos_size + query_size])
        support_pos1, query_pos1, _ = np.split(self.data_pos1[indices], [support_pos_size, support_pos_size + query_size])
        support_pos2, query_pos2, _ = np.split(self.data_pos2[indices], [support_pos_size, support_pos_size + query_size])
        support_mask, query_mask, _ = np.split(self.data_mask[indices], [support_pos_size, support_pos_size + query_size])
        support_id, query_id, _ = np.split(self.uid[indices], [support_pos_size, support_pos_size + query_size])
        support_entpair = list(self.data_entpair[indices[:support_pos_size]])
        query['entpair'] += list(self.data_entpair[indices[support_pos_size:]])

        # support_neg = train_data_loader.next_batch(support_pos_size * support_neg_rate)
        # support_neg['label'] = np.zeros((support_neg_rate * support_pos_size), dtype=np.int32)
        
        if neg_train_loader is None:
            neg_train_loader = train_data_loader
        for i, class_name in enumerate(neg_train_loader.rel2scope.keys()[:support_neg_rate]):
            scope = neg_train_loader.rel2scope[class_name]
            support_neg['word'].append(neg_train_loader.data_word[scope[0]:scope[0]+support_pos_size])  
            support_neg['pos1'].append(neg_train_loader.data_pos1[scope[0]:scope[0]+support_pos_size])    
            support_neg['pos2'].append(neg_train_loader.data_pos2[scope[0]:scope[0]+support_pos_size])    
            support_neg['mask'].append(neg_train_loader.data_mask[scope[0]:scope[0]+support_pos_size])
            support_neg['id'].append(neg_train_loader.uid[scope[0]:scope[0]+support_pos_size])
            support_neg['label'] += [0] * support_pos_size

        support_neg['word'] = np.concatenate(support_neg['word'], 0)
        support_neg['pos1'] = np.concatenate(support_neg['pos1'], 0)
        support_neg['pos2'] = np.concatenate(support_neg['pos2'], 0)
        support_neg['mask'] = np.concatenate(support_neg['mask'], 0)
        support_neg['id'] = np.concatenate(support_neg['id'], 0)
        support_neg['label'] = np.array(support_neg['label'])

        support_pos['word'] = support_word
        support_pos['pos1'] = support_pos1
        support_pos['pos2'] = support_pos2
        support_pos['mask'] = support_mask
        support_pos['label'] = np.ones((support_pos_size), dtype=np.int32)
        support_pos['id'] = support_id
        support_pos['entpair'] = support_entpair

        query['word'].append(query_word)
        query['pos1'].append(query_pos1) 
        query['pos2'].append(query_pos2)
        query['mask'].append(query_mask)
        query['id'].append(query_id)
        query['label'] += [1] * query_size

        # Other query classes (negative)
        if use_train_neg:
            neg_loader = train_data_loader
            target_classes = random.sample(neg_loader.rel2scope.keys(), query_class) # discard 0 class 
        else:
            neg_loader = self

        for i, class_name in enumerate(target_classes):
            if class_name == main_class:
                continue
            scope = neg_loader.rel2scope[class_name]
            query['word'].append(neg_loader.data_word[scope[0]+support_pos_size:scope[0]+support_pos_size+query_size])  
            query['pos1'].append(neg_loader.data_pos1[scope[0]+support_pos_size:scope[0]+support_pos_size+query_size])    
            query['pos2'].append(neg_loader.data_pos2[scope[0]+support_pos_size:scope[0]+support_pos_size+query_size])    
            query['mask'].append(neg_loader.data_mask[scope[0]+support_pos_size:scope[0]+support_pos_size+query_size])
            query['id'].append(neg_loader.uid[scope[0]+support_pos_size:scope[0]+support_pos_size+query_size])
            query['label'] += [0] * query_size
            query['entpair'] += list(neg_loader.data_entpair[scope[0]+support_pos_size:scope[0]+support_pos_size+query_size])

        query['word'] = np.concatenate(query['word'], 0)
        query['pos1'] = np.concatenate(query['pos1'], 0)
        query['pos2'] = np.concatenate(query['pos2'], 0)
        query['mask'] = np.concatenate(query['mask'], 0)
        query['id'] = np.concatenate(query['id'], 0)
        query['label'] = np.array(query['label'])

        support_pos['word'] = Variable(torch.from_numpy(support_pos['word']).long()) 
        support_pos['pos1'] = Variable(torch.from_numpy(support_pos['pos1']).long())
        support_pos['pos2'] = Variable(torch.from_numpy(support_pos['pos2']).long())
        support_pos['mask'] = Variable(torch.from_numpy(support_pos['mask']).long())
        support_pos['id'] = Variable(torch.from_numpy(support_pos['id']).long())
        support_pos['label'] = Variable(torch.from_numpy(support_pos['label']).long())

        support_neg['word'] = Variable(torch.from_numpy(support_neg['word']).long()) 
        support_neg['pos1'] = Variable(torch.from_numpy(support_neg['pos1']).long())
        support_neg['pos2'] = Variable(torch.from_numpy(support_neg['pos2']).long())
        support_neg['mask'] = Variable(torch.from_numpy(support_neg['mask']).long())
        support_neg['id'] = Variable(torch.from_numpy(support_neg['id']).long())
        support_neg['label'] = Variable(torch.from_numpy(support_neg['label']).long())

        query['word'] = Variable(torch.from_numpy(query['word']).long()) 
        query['pos1'] = Variable(torch.from_numpy(query['pos1']).long())
        query['pos2'] = Variable(torch.from_numpy(query['pos2']).long())
        query['mask'] = Variable(torch.from_numpy(query['mask']).long())
        query['id'] = Variable(torch.from_numpy(query['id']).long())
        query['label'] = Variable(torch.from_numpy(query['label']).long())

        # To cuda
        if self.cuda:
            for key in ['word', 'pos1', 'pos2', 'mask', 'label', 'id']:
                support_pos[key] = support_pos[key].cuda()
                support_neg[key] = support_neg[key].cuda()
                query[key] = query[key].cuda()

        return support_pos, support_neg, query, main_class

    def next_fewshot_one(self, N, K, Q):
        target_classes = random.sample(self.rel2scope.keys(), N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            id = self.uid[indices]
            support_word, query_word, _ = np.split(word, [K, K + Q])
            support_pos1, query_pos1, _ = np.split(pos1, [K, K + Q])
            support_pos2, query_pos2, _ = np.split(pos2, [K, K + Q])
            support_mask, query_mask, _ = np.split(mask, [K, K + Q])
            support_id, query_id, _ = np.split(id, [K, K + Q])
            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            support_set['id'].append(support_id)
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_set['id'].append(query_id)
            query_label += [i] * Q

        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        support_set['id'] = np.stack(support_set['id'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_set['id'] = np.concatenate(query_set['id'], 0)
        query_label = np.array(query_label)

        perm = np.random.permutation(N * Q)
        query_set['word'] = query_set['word'][perm]
        query_set['pos1'] = query_set['pos1'][perm]
        query_set['pos2'] = query_set['pos2'][perm]
        query_set['mask'] = query_set['mask'][perm]
        query_set['id'] = query_set['id'][perm]
        query_label = query_label[perm]

        return support_set, query_set, query_label

    def next_fewshot_batch(self, B, N, K, Q):
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
        label = []
        for one_sample in range(B):
            current_support, current_query, current_label = self.next_one(N, K, Q)
            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            support['id'].append(current_support['id'])
            query['word'].append(current_query['word'])
            query['pos1'].append(current_query['pos1'])
            query['pos2'].append(current_query['pos2'])
            query['id'].append(current_query['id'])
            label.append(current_label)
        support['word'] = Variable(torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length))
        support['pos1'] = Variable(torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length)) 
        support['pos2'] = Variable(torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length)) 
        support['mask'] = Variable(torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)) 
        support['id'] = Variable(torch.from_numpy(np.stack(support['id'], 0)).long().view(-1, self.max_length)) 
        query['word'] = Variable(torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)) 
        query['pos1'] = Variable(torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length)) 
        query['pos2'] = Variable(torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length)) 
        query['mask'] = Variable(torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)) 
        query['id'] = Variable(torch.from_numpy(np.stack(query['id'], 0)).long().view(-1, self.max_length)) 
        label = Variable(torch.from_numpy(np.stack(label, 0).astype(np.int64)).long())
        
        # To cuda
        if self.cuda:
            for key in support:
                support[key] = support[key].cuda()
            for key in query:
                query[key] = query[key].cuda()
            label = label.cuda()

        return support, query, label
