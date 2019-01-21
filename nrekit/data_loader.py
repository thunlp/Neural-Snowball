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

    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=False, cuda=True, distant=False):
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
            self.rel2id = {}
            self.rel_tot = 0
            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
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

        self.index = list(range(self.instance_tot))
        random.shuffle(self.index)
        self.current = 0

    def next_batch(self, batch_size):
        batch = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        if self.current + batch_size > len(self.index):
            self.index = list(range(self.instance_tot))
            random.shuffle(self.index)
            self.current = 0
        current_index = self.index[self.current:self.current+batch_size]
        self.current += batch_size

        batch['word'] = Variable(torch.from_numpy(self.data_word[current_index]).long()) 
        batch['pos1'] = Variable(torch.from_numpy(self.data_pos1[current_index]).long())
        batch['pos2'] = Variable(torch.from_numpy(self.data_pos2[current_index]).long())
        batch['mask'] = Variable(torch.from_numpy(self.data_mask[current_index]).long())
        batch['label']= Variable(torch.from_numpy(self.data_label[current_index]).long())

        # To cuda
        if self.cuda:
            for key in batch:
                batch[key] = batch[key].cuda()

        return batch

    def next_multi_class(self, num_size, num_class):
        '''
        num_size: The num of instances for ONE class. The total size is num_size * num_classes.
        num_class: The num of classes (include the positive class).
        '''
        target_classes = random.sample(self.rel2scope.keys(), num_class)
        batch = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), num_size, False)
            batch['word'].append(self.data_word[indices])
            batch['pos1'].append(self.data_pos1[indices])
            batch['pos2'].append(self.data_pos2[indices])
            batch['mask'].append(self.data_mask[indices])

        batch['word'] = np.concatenate(batch['word'], 0)
        batch['pos1'] = np.concatenate(batch['pos1'], 0)
        batch['pos2'] = np.concatenate(batch['pos2'], 0)
        batch['mask'] = np.concatenate(batch['mask'], 0)

        batch['word'] = Variable(torch.from_numpy(batch['word']).long()) 
        batch['pos1'] = Variable(torch.from_numpy(batch['pos1']).long())
        batch['pos2'] = Variable(torch.from_numpy(batch['pos2']).long())
        batch['mask'] = Variable(torch.from_numpy(batch['mask']).long())

        # To cuda
        if self.cuda:
            for key in batch:
                batch[key] = batch[key].cuda()

        return batch

    def next_new_relation(self, train_data_loader, support_size, query_size, unlabelled_size, query_class, negative_rate=5):
        '''
        support_size: The num of instances for positive / negative. The total support size is support_size * 2.
        query_size: The num of instances for ONE class in query set. The total query size is query_size * query_class.
        query_class: The num of classes in query (include the positive class).
        '''
        target_classes = random.sample(self.rel2scope.keys(), query_class) # 0 class is the new relation 
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'label': []}
        unlabelled_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        # New relation
        scope = self.rel2scope[target_classes[0]]
        indices = np.random.choice(list(range(scope[0], scope[1])), support_size + query_size + unlabelled_size, False)
        support_word, query_word, unlabelled_word, _ = np.split(self.data_word[indices], [support_size, support_size + query_size, support_size + query_size + unlabelled_size])
        support_pos1, query_pos1, unlabelled_pos1, _ = np.split(self.data_pos1[indices], [support_size, support_size + query_size, support_size + query_size + unlabelled_size])
        support_pos2, query_pos2, unlabelled_pos2, _ = np.split(self.data_pos2[indices], [support_size, support_size + query_size, support_size + query_size + unlabelled_size])
        support_mask, query_mask, unlabelled_mask, _ = np.split(self.data_mask[indices], [support_size, support_size + query_size, support_size + query_size + unlabelled_size])

        negative_support = train_data_loader.next_batch(support_size * negative_rate)
        support_set['word'] = np.concatenate((support_word, negative_support['word']), 0)
        support_set['pos1'] = np.concatenate((support_pos1, negative_support['pos1']), 0)
        support_set['pos2'] = np.concatenate((support_pos2, negative_support['pos2']), 0)
        support_set['mask'] = np.concatenate((support_mask, negative_support['mask']), 0)
        support_set['label'] = np.concatenate((np.ones((support_size), dtype=np.int32), np.zeros((negative_rate * support_size), dtype=np.int32)), 0)

        query_set['word'].append(query_word)
        query_set['pos1'].append(query_pos1) 
        query_set['pos2'].append(query_pos2)
        query_set['mask'].append(query_mask)
        query_set['label'] += [1] * query_size

        unlabelled_set['word'].append(unlabelled_word)
        unlabelled_set['pos1'].append(unlabelled_pos1) 
        unlabelled_set['pos2'].append(unlabelled_pos2)
        unlabelled_set['mask'].append(unlabelled_mask)

        # Other query classes (negative)
        for i, class_name in enumerate(target_classes[1:]):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), query_size + unlabelled_size, False)
            query_word, unlabelled_word, _ = np.split(self.data_word[indices], [query_size, query_size + unlabelled_size])  
            query_pos1, unlabelled_pos1, _ = np.split(self.data_pos1[indices], [query_size, query_size + unlabelled_size])    
            query_pos2, unlabelled_pos2, _ = np.split(self.data_pos2[indices], [query_size, query_size + unlabelled_size])    
            query_mask, unlabelled_mask, _ = np.split(self.data_mask[indices], [query_size, query_size + unlabelled_size])    

            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_set['label'] += [0] * query_size

            unlabelled_set['word'].append(unlabelled_word)
            unlabelled_set['pos1'].append(unlabelled_pos1)
            unlabelled_set['pos2'].append(unlabelled_pos2)
            unlabelled_set['mask'].append(unlabelled_mask)

        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_set['label'] = np.array(query_set['label'])

        unlabelled_set['word'] = np.concatenate(unlabelled_set['word'], 0)
        unlabelled_set['pos1'] = np.concatenate(unlabelled_set['pos1'], 0)
        unlabelled_set['pos2'] = np.concatenate(unlabelled_set['pos2'], 0)
        unlabelled_set['mask'] = np.concatenate(unlabelled_set['mask'], 0)

        support_set['word'] = Variable(torch.from_numpy(support_set['word']).long()) 
        support_set['pos1'] = Variable(torch.from_numpy(support_set['pos1']).long())
        support_set['pos2'] = Variable(torch.from_numpy(support_set['pos2']).long())
        support_set['mask'] = Variable(torch.from_numpy(support_set['mask']).long())
        support_set['label'] = Variable(torch.from_numpy(support_set['label']).long())

        query_set['word'] = Variable(torch.from_numpy(query_set['word']).long()) 
        query_set['pos1'] = Variable(torch.from_numpy(query_set['pos1']).long())
        query_set['pos2'] = Variable(torch.from_numpy(query_set['pos2']).long())
        query_set['mask'] = Variable(torch.from_numpy(query_set['mask']).long())
        query_set['label'] = Variable(torch.from_numpy(query_set['label']).long())

        unlabelled_set['word'] = Variable(torch.from_numpy(unlabelled_set['word']).long()) 
        unlabelled_set['pos1'] = Variable(torch.from_numpy(unlabelled_set['pos1']).long())
        unlabelled_set['pos2'] = Variable(torch.from_numpy(unlabelled_set['pos2']).long())
        unlabelled_set['mask'] = Variable(torch.from_numpy(unlabelled_set['mask']).long())
 
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
        batch['pos1'] = Variable(torch.from_numpy(self.data_pos1[scope]).long())
        batch['pos2'] = Variable(torch.from_numpy(self.data_pos2[scope]).long())
        batch['mask'] = Variable(torch.from_numpy(self.data_mask[scope]).long())
        batch['label']= Variable(torch.from_numpy(self.data_label[scope]).long())
        batch['id']   = scope
        batch['entpair'] = entpair * len(scope)

        # To cuda
        if self.cuda:
            for key in ['word', 'pos1', 'pos2', 'mask']:
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
            candidate['id'] += list(indices)
            candidate['entpair'] += list(self.data_entpair[indices])

        candidate['word'] = np.concatenate(candidate['word'], 0)
        candidate['pos1'] = np.concatenate(candidate['pos1'], 0)
        candidate['pos2'] = np.concatenate(candidate['pos2'], 0)
        candidate['mask'] = np.concatenate(candidate['mask'], 0)

        candidate['word'] = Variable(torch.from_numpy(candidate['word']).long()) 
        candidate['pos1'] = Variable(torch.from_numpy(candidate['pos1']).long())
        candidate['pos2'] = Variable(torch.from_numpy(candidate['pos2']).long())
        candidate['mask'] = Variable(torch.from_numpy(candidate['mask']).long())

        # To cuda
        if self.cuda:
            for key in ['word', 'pos1', 'pos2', 'mask']:
                candidate[key] = candidate[key].cuda()

        return candidate
   
    def get_one_new_relation(self, train_data_loader, support_pos_size, support_neg_rate, query_size, query_class):
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
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'label': []}

        # New relation
        scope = self.rel2scope[target_classes[0]]
        indices = np.random.choice(list(range(scope[0], scope[1])), support_pos_size + query_size, False)
        support_word, query_word, _ = np.split(self.data_word[indices], [support_pos_size, support_pos_size + query_size])
        support_pos1, query_pos1, _ = np.split(self.data_pos1[indices], [support_pos_size, support_pos_size + query_size])
        support_pos2, query_pos2, _ = np.split(self.data_pos2[indices], [support_pos_size, support_pos_size + query_size])
        support_mask, query_mask, _ = np.split(self.data_mask[indices], [support_pos_size, support_pos_size + query_size])
        support_id = list(indices[:support_pos_size])
        support_entpair = list(self.data_entpair[indices[:support_pos_size]])

        support_neg = train_data_loader.next_batch(support_pos_size * support_neg_rate)
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
        query['label'] += [1] * query_size

        # Other query classes (negative)
        for i, class_name in enumerate(target_classes[1:]):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), query_size, False)
            query['word'].append(self.data_word[indices])  
            query['pos1'].append(self.data_pos1[indices])    
            query['pos2'].append(self.data_pos2[indices])    
            query['mask'].append(self.data_mask[indices])
            query['label'] += [0] * query_size

        query['word'] = np.concatenate(query['word'], 0)
        query['pos1'] = np.concatenate(query['pos1'], 0)
        query['pos2'] = np.concatenate(query['pos2'], 0)
        query['mask'] = np.concatenate(query['mask'], 0)
        query['label'] = np.array(query['label'])

        support_pos['word'] = Variable(torch.from_numpy(support_pos['word']).long()) 
        support_pos['pos1'] = Variable(torch.from_numpy(support_pos['pos1']).long())
        support_pos['pos2'] = Variable(torch.from_numpy(support_pos['pos2']).long())
        support_pos['mask'] = Variable(torch.from_numpy(support_pos['mask']).long())
        support_pos['label'] = Variable(torch.from_numpy(support_pos['label']).long())
        support_neg['label'] = Variable(torch.from_numpy(support_neg['label']).long())

        query['word'] = Variable(torch.from_numpy(query['word']).long()) 
        query['pos1'] = Variable(torch.from_numpy(query['pos1']).long())
        query['pos2'] = Variable(torch.from_numpy(query['pos2']).long())
        query['mask'] = Variable(torch.from_numpy(query['mask']).long())
        query['label'] = Variable(torch.from_numpy(query['label']).long())

        # To cuda
        if self.cuda:
            for key in ['word', 'pos1', 'pos2', 'mask', 'label']:
                support_pos[key] = support_pos[key].cuda()
            for key in ['word', 'pos1', 'pos2', 'mask', 'label']:
                query[key] = query[key].cuda()
            support_neg[key] = support_neg[key].cuda()

        return support_pos, support_neg, query, target_classes[0]

