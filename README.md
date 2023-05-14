# StoryTrans

This repo inculdes the code in the paper [StoryTrans: Non-Parallel Story Author-Style Transfer with Discourse Representations and Content Enhancing](https://arxiv.org/abs/2208.13423) (ACL 2023 Long Paper).

StoryTrans leverages discourse representations to capture source content information and transfer them to target styles with learnable style embeddings. 

![Main_figure](figure/main_figure.png)

## Prerequisites

The prerequisites for running the code are listed in the `requirement.txt`. Make sure you have the necessary environment and dependencies set up before proceeding with the installation and execution of the code.

To install the required dependencies, you can use the following command:

```bash
conda create --name <env> --file <this file>
```

## Quick Start

#### 1. Training of Discourse Representation Transfer

Execute the following command to train for first stage: 
```shell
sh text_style_transfer/StyTrans_tran.sh
```

#### 2. Training of Content Preservation Enhancing
Execute the following command to train for second stage: 
```shell
sh text_style_transfer/MaskFill_train.sh
```

#### 3. Generation 
Execute the following command to generate your style transfer texts: 
```shell
sh text_style_transfer/StyTrans_stage_1_test_zh.sh
sh text_style_transfer/MaskFill_gen_zh.sh
```

### Citation

Please kindly cite our paper if this paper and the code are helpful.

```
@article{zhu2022storytrans,
  title={StoryTrans: Non-Parallel Story Author-Style Transfer with Discourse Representations and Content Enhancing},
  author={Zhu, Xuekai and Guan, Jian and Huang, Minlie and Liu, Juan},
  journal={arXiv preprint arXiv:2208.13423},
  year={2022}
}
```