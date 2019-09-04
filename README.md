
# RNS

This repository is the implementation of RNS ([arXiv](https://arxiv.org/abs/1907.00590)): 

> A Review-Driven Neural Model for Sequential Recommendation (IJCAI 2019)
Chenliang Li, Xichuan Niu, Xiangyang Luo, Zhenzhong Chen, Cong Quan

## Requirements
* Python 3.6
* PyTorch 0.4
* Numpy
* Pandas
* SciPy

## Files in the folder
- `data/`
  - `reviews_Amazon_Instant_Video.json/`
    - `video_train.csv`: csv file (user_id, item_id, rating, timestamp) for training
    - `video_test.csv`: csv file (user_id, item_id, rating, timestamp) for testing
    - `vocabulary`: vocabulary of user reviews text
    - `u_text`: review documents written by user u
    - `i_text`: review documents written for item i

## Running the code

1. Install required packages.
2. run `python train_model.py`

## Citation

Please cite our paper if you find this code useful for your research:

```
@inproceedings{RNS2019,
  author    = {Chenliang Li and
               Xichuan Niu and
               Xiangyang Luo and
               Zhenzhong Chen and
               Cong Quan},
  title     = {A Review-Driven Neural Model for Sequential Recommendation},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI} 2019},
  pages     = {2866--2872},
  year      = {2019}
}
```

## Acknowledgment

This source code is built on top of [caser_pytorch](https://github.com/graytowne/caser_pytorch). We thank the author for his amazing work.
