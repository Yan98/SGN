# Spatial Transcriptomics Analysis of Zero-shot Gene Expression Prediction

[This repository](https://github.com/noxsine/LDP) contains the PyTorch code for our paper "Spatial Transcriptomics Analysis of Zero-shot Gene Expression Prediction".

> [paper]() | [arxiv](https://arxiv.org/pdf/2401.14772)

**The code will come soon!**


## Introduction
Spatial transcriptomics (ST) captures gene expression fine-grained distinct regions (i.e., windows) of a tissue slide. Traditional supervised learning frameworks applied to model ST are constrained to predicting expression of gene types seen during training from slide image windows, failing to generalize to unseen gene types. To overcome this limitation, we propose a semantic guided network, a pioneering zero- shot gene expression prediction framework. Considering a gene type can be described by functionality and phenotype, we dynamically embed a gene type to a vector per its functionality and phenotype, and employ this vector to project slide image windows to gene expression in feature space, unleashing zero-shot expression prediction for unseen gene types. The gene type functionality and phenotype are queried with a carefully designed prompt from a pre-trained large language model. On standard benchmark datasets, we demonstrate competitive zero-shot performance compared to past state-of-the-art supervised learning approaches.

<div align=center>
<img src="asset/intro.png", width=500/>
</div>

## Framework

<div align=center>
<img src="asset/model.png", width=500/>
</div>

## Requirements

Please refer to [requirements.txt](./requirements.txt).

## How to run

```bash
python main.py
```


## Citation

```

@article{yang2024spatialtranscriptomicsanalysiszeroshot,
      title={Spatial Transcriptomics Analysis of Zero-shot Gene Expression Prediction}, 
      author={Yan Yang and Md Zakir Hossain and Xuesong Li and Shafin Rahman and Eric Stone},
      year={2024},
      eprint={2401.14772},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.14772}, 
}
