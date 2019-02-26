import numpy as np
import json

train_train = json.load(open('./train_train.json'))
train_val = json.load(open('./train_val.json'))
val = json.load(open('./val.json'))
test = json.load(open('./test.json'))
distant = json.load(open('./distant.json'))

total = 0

for data, name in [(train_train, 'train_train'), (train_val, 'train_val'), (val, 'val'), (test, 'test'), (distant, 'distant')]:
    print(name)
    count = 0
    for rel in data:
        count += len(data[rel])
    data_id = np.array(list(range(total, total + count)), dtype=np.int32)
    np.save(name + '_uid.npy', data_id)
    total += count

