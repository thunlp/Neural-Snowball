import numpy as np

for encoder in ['cnn']:
    print(encoder)
    for module in ['encoder', 'siamese']:
        print(module)
        merge = []
        for dataset in ['train_train', 'train_val', 'val', 'test', 'distant']:
            print(dataset)
            _ = np.load('_'.join([encoder, module, 'on_fewrel']) + '.' + dataset + '.npy')
            merge.append(_)
        merge = np.concatenate(merge, 0)
        np.save('../_repre/' + '_'.join([encoder, module, 'on_fewrel']) + '.npy', merge)
