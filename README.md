## Attention-augmented multi-domain cooperative graph representation learning for molecular interaction prediction

This repository contains the source code and benchmark datasets used in this paper.

## Introduction

Motivation:&#x20;

Accurate identification of molecular interactions is crucial for biological network analysis, which can provide valuable insights into fundamental life regulatory mechanisms. Despite considerable progress driven by computational technologies, these methods typically depend on the task-specific prior knowledge or inherent structural properties of molecules, limiting their generalizability and applicability. Recently, graph-based methods have emerged as an appropriate and prevalent approach for predicting links in molecular networks, but most of them primarily focus on aggregating topological information within single domains, resulting in inadequate characterization.

Results:

In this paper, a generalized attention-augmented multi-domain cooperative graph representation learning framework, named AMCGRL, is proposed for multifarious molecular interaction prediction tasks. Concretely, AMCGRL incorporates multiple graph encoders to simultaneously learn molecular representations from both intra-domain and inter-domain graphs in a comprehensive perspective. Moreover, the cross-domain decoder is designed as the bridge between graph encoders to parse out complementary information across diverse domains. Furthermore, a hierarchical mutual attention mechanism is developed to explore complicated pair-wise interaction patterns between different types of molecules through inter-molecule communicative learning. Extensive experiments conducted on the benchmark datasets demonstrate the superior representation learning capability of AMCGRL compared to the state-of-the-art methods, proving its efficacy in facilitating molecular interaction prediction.



## Dataset

In this paper, three different types of biological networks are collected from previous work or publicly available databases for the molecular interaction prediction task.

*   **ATH-PepPI** comprises 3*, *966 interactions between 243 peptides and 1*, *039 proteins in the *Arabidopsis thaliana *(*A. thaliana*).

*   **ATH-MLI** consists of 340 miRNAs and 516 lncRNAs, with a total of 3*, *099 interactions between these two classes of non-coding RNAs in the *A. thaliana*.

*   **HS-RPI** consists of 7*, *317 interactions of *Homo sapiens *(*H. sapiens*) between 1*, *874 RNAs and 118 proteins.

## Setup and dependencies

*   Python 3.7

*   Torch 1.10.0

*   Torch\_geometric 2.1.0&#x20;

*   Numpy 1.24.4&#x20;

*   Scikit-learn 1.3.2

## Code details

*   AMCGRL-AA.py: train and evaluate the model.

*   Model.py: AMCGRL modules.

*   load\_data.py: data importing and processing.

*   test\_scores.py: performance calculation.

```python
python AMCGRL-AA.py
```

## Citation

Zhaowei Wang, Jun Meng, Haibin Li, et al. "Attention-augmented multi-domain cooperative graph representation learning for molecular interaction prediction."   ***Neural Networks*** (2024) \[*Under Review*]
