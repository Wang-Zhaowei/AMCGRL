# AMCGRL
The source code and datasets used in "Attention-augmented multi-domain cooperative graph representation learning for molecular interaction prediction".

## Introduction

### Motivation:

Accurate identification of molecular interactions is crucial for biological network analysis, which can provide valuable insights into fundamental regulatory mechanisms. Despite considerable progress driven by computational advancements, existing methods often rely on task-specific prior knowledge or inherent structural properties of molecules, which limits their generalizability and applicability. Recently, graph-based methods have emerged as a promising approach for predicting links in molecular networks. However, most of these methods focus primarily on aggregating topological information within individual domains, leading to an inadequate characterization of molecular interactions.

### Results:

we propose AMCGRL, a generalized multi-domain cooperative graph representation learning framework for multifarious molecular interaction prediction tasks. Concretely, AMCGRL incorporates multiple graph encoders to simultaneously learn molecular representations from both intra-domain and inter-domain graphs in a comprehensive manner. Then, the cross-domain decoder is employed to bridge these graph encoders to facilitate the extraction of task-relevant information across different domains. Additionally, a hierarchical mutual attention mechanism is developed to capture complex pairwise interaction patterns between distinct types of molecules through inter-molecule communicative learning. Extensive experiments conducted on the various datasets demonstrate the superior representation learning capability of AMCGRL compared to the state-of-the-art methods, proving its effectiveness in advancing the prediction of molecular interactions.



## Dataset

In this paper, different types of biological networks are collected from previous work or publicly available databases for the molecular interaction prediction task.

*   **PepPI3966** comprises 3,966 interactions between 243 peptides and 1,039 proteins in the *Arabidopsis thaliana* (*A. thaliana*).

*   **MLI3099** consists of 340 miRNAs and 516 lncRNAs, with a total of 3,099 interactions between these two classes of non-coding RNAs in the *A. thaliana*.

*   **RPI7317** consists of 7,317 interactions of *Homo sapiens* (*H. sapiens*) between 1,874 lncRNAs and 118 proteins.
*   **Yamanishi's dataset** contains four subsets of protein families: (1) Enzymes, (2) Ion channels, (3) G-Protein-Coupled Receptors (GPCRs), and (4) Nuclear receptors, which can be available at [Yamanishi's Supplements](https://members.cbio.mines-paristech.fr/~yyamanishi/pharmaco/https://members.cbio.mines-paristech.fr/~yyamanishi/pharmaco/).

## Setup and dependencies

*   Python 3.7

*   Torch 1.10.0

*   Torch\_geometric 2.1.0&#x20;

*   Numpy 1.24.4&#x20;

*   Scikit-learn 1.3.2

## Code details

*   main.py: train and evaluate the model.

*   Model.py: AMCGRL modules.

*   load\_data.py: data importing and processing.

*   test\_scores.py: performance calculation.

To run the AMCGRL framework, you can set the type of molecular interaction targeted, such as "**N**" denoting nucleotide sequence based molecules, and "**A**" represents amino acid sequence based molecules. Then, the experimental results are saved in the "/Results/**out_file**" file.
```python
python main.py --inter_type NN --input_x_file miRNAs.txt --input_y_file lncRNAs.txt --out_file results.txt
```

## Citation

Zhaowei Wang, Jun Meng, Haibin Li, et al. "Attention-augmented multi-domain cooperative graph representation learning for molecular interaction prediction."   ***Neural Networks*** (2024) \[*Under Review*]
