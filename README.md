# DAMS

Pytorch implementation of the ACL-2022 (findings) paper: [Divide and Conquer: Text Semantic Matching with Disentangled Keywords and Intents](https://arxiv.org/abs/2203.02898).

## Environments

* Python 3.9.7
 
* pytorch 1.10.2

* transformers 4.17.0

* datasets 2.0.0

* RTX 3090 GPU & Titan RTX

* CUDA 11.4

## Data

All the processed datastes used in our work are available at [Google Drive](https://drive.google.com/file/d/1OugsTLxqdoxAWaC93hn8xAoiMzjdxjXg/view?usp=sharing) or [Baidu Pan  (extract code: w2op)](https://pan.baidu.com/s/1H1fpJJL9wZEMDicdBOemwg?pwd=w2op), including QQP, MRPC, and Medical-SM.

## Usage

* Download raw datasets from the above data links and put them into the directory **raw_data** like this:

	```
	--- raw_data
	  |
	  |--- medical
	  |
      |--- mrpc
      |
      |--- qqp
	```

* We have tried various pre-trained models. The following models work fine with our code:

    * model names for QQP and MRPC:
        - roberta-base
        - roberta-large
        - bert-base-uncased
        - bert-large-uncased
        - albert-base-v2
        - albert-large-v2
        - microsoft/deberta-large
        - microsoft/deberta-base
        - funnel-transformer/medium

    * model names for Medical:
        - hfl/chinese-macbert-base
        - hfl/chinese-macbert-large
        - hfl/chinese-roberta-wwm-ext
        - hfl/chinese-roberta-wwm-ext-large

* Pre-process datasets.

    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path raw_data/mrpc
    ```
    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path raw_data/qqp
    ```
    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path raw_data/medical
    ```

* Training and Evaluation (Baseline)

    * MRPC
    ```
    PYTHONPATH=. python -u src/main.py \
        -baseline \
        -task mrpc \
        -model roberta-large \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 1 \
        -test_batch_size 128 \
        >> logs/mrpc.roberta_large.baseline.log
    ```

    * QQP
    ```
    PYTHONPATH=. python -u src/main.py \
        -baseline \
        -task qqp \
        -model roberta-large \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 4 \
        -test_batch_size 128 \
        >> logs/qqp.roberta_large.baseline.log
    ```

    * Medical
    ```
    PYTHONPATH=. python -u src/main.py \
        -baseline \
        -task medical \
        -model hfl/chinese-roberta-wwm-ext-large \
        -num_labels 3 \
        -batch_size 16 \
        -accum_count 4 \
        -test_batch_size 128 \
        >> logs/medical.roberta_large.baseline.log
    ```

* Training and Evaluation (DC-Match)

    * MRPC
    ```
    PYTHONPATH=. python -u src/main.py \
        -task mrpc \
        -model roberta-large \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 1 \
        -test_batch_size 128 \
        >> logs/mrpc.roberta_large.log
    ```

    * QQP
    ```
    PYTHONPATH=. python -u src/main.py \
        -task qqp \
        -model roberta-large \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 4 \
        -test_batch_size 128 \
        >> logs/qqp.roberta_large.log
    ```

    * Medical
    ```
    PYTHONPATH=. python -u src/main.py \
        -task medical \
        -model hfl/chinese-roberta-wwm-ext-large \
        -num_labels 3 \
        -batch_size 16 \
        -accum_count 4 \
        -test_batch_size 128 \
        >> logs/medical.roberta_large.log
    ```

## Citation

    @article{zou2022divide,
             title={Divide and Conquer: Text Semantic Matching with Disentangled Keywords and Intents},
             author={Zou, Yicheng and Liu, Hongwei and Gui, Tao and Wang, Junzhe and Zhang, Qi and Tang, Meng and Li, Haixiang and Wang, Daniel},
             journal={arXiv preprint arXiv:2203.02898},
             year={2022}
    }
