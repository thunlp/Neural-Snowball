# Neural Snowball for Few-Shot Relation Learning

Code and dataset of AAAI2020 Paper **Neural Snowball for Few-Shot Relation Learning**. [paper](https://arxiv.org/pdf/1908.11007.pdf)

## Citation

Please cite our paper if you find it helpful.

```
@inproceedings{gao2020neural,
    title = "Neural Snowball for Few-Shot Relation Learning",
    author = "Gao, Tianyu and Han, Xu and Xie, Ruobing and Liu, Zhiyuan and Lin, Fen and Lin, Leyu and Sun, Maosong",
    booktitle = "Proceedings of AAAI",
    year = "2020"
}
```

## Dataset

Please download `data.tar` from [Aliyun](https://thunlp.oss-cn-qingdao.aliyuncs.com/gaotianyu/neural_snowball/data.tar) and unzip it as the `data` folder:

```bash
tar xvf data.tar
```

Note that the test set of FewRel is held for online evaluation, so we do not provide the test set here. You can get almost the same results by using the val set of FewRel.

## Requirements

This repository has been tested with `Python 3.6`, `torch==1.3.0` and `transformers==2.2.1`.

## Get Started

**Step 1** Pre-train the text encoder.

For CNN, using the following command,
```
python train_cnn_encoder.py
```

and for BERT, using the following command.
```
python train_bert_encoder.py
```

**Step 2** Pre-train the relational siamese network.

For CNN, using the following command,
```
python train_cnn_siamese.py
```

and for BERT, using the following command.
```
python train_bert_siamese.py
```

**Step 3** Start neural snowball

For CNN, using the following command,
```
python test_cnn_snowball.py --shot 5 --eval_iter 1000
```

where `--shot` indicates the number of the starting seeds (instances), and `--eval_iter` indicates the eval iteration. The larger `eval_iter` is, the more accurate results you can get.

For BERT, using
```
python test_bert_snowball.py --shot 5 --eval_iter 1000
```

During evaluation, you will see something like:

```
[EVAL] step:  100 | f1: 0.4282, prec: 50.40%, recall: 44.16% | [baseline] f1: 0.2197, prec: 49.93%, rec: 15.98%
```

The left part is the accumulated result of neural snowball, and the right part is the fine-tuning baseline result.

At the end, you will get a final result with the order `{BASELINE_F1} {BASELINE_P} {BASELINE_R} {SNOWBALL_F1} {SNOWBALL_P} {SNOWBALL_R}`.
