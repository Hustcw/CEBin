<h1 align="center">CEBin: A Cost-Effective Framework for Large-Scale Binary Code Similarity Detection</h1>

<h4 align="center">
<p>
<a href=#about>About</a> |
<a href=#news>News</a> |
<a href=#quickstart>QuickStart</a> |
<a href=#details>Details</a> |
<a href=#citation>Citation</a>
</p>
</h4>

## About
CEBin is a cost-effective framework designed for large-scale binary code similarity detection. CEBin achieves a strong balance between accuracy and computational efficiency, making it practical for industrial-scale binary analysis tasks. The framework includes both **embedding-based retrieval** and **comparison-based classification** models.  
We release datasets, pretrained models, and tokenizers to facilitate research and application in the binary code analysis community.

Dataset and model download link: [Download Here](https://cloud.vul337.team:8443/s/SNREWFrYeYnnzd8)

## News

- [2025/4/21] Released CEBin datasets, tokenizers, and pretrained models to the public.
- [2024/3/02] CEBin paper is accepted by ISSTA 2024.
- [2024/2/29] CEBin paper is available online (https://arxiv.org/abs/2402.18818)

## QuickStart
Please refer to `finetune/demo_embedding.py` and `finetune/demo_comparison.py`.

## Details
This repository contains the following key materials:

1. **Models**:
   - `CEBin-Embedding-Cisco.bin`: Embedding model for vector retrieval.
   - `CEBin-Comparison-Cisco.bin`: Comparison model for direct pair classification.

2. **Tokenizer**:
   - `cebin-tokenizer/`: Pretrained tokenizer specialized for binary function representation.

3. **Datasets**:
   - `BinaryCorp`,`Cisco`,`Trex`: Large-scale dataset containing diversified architectures and optimization levels processed by BinaryNinja.

4. **Code**
   - pretraining and finetuning scripts for training CEBin models

### Supported Architectures
- x86, x86-64
- ARM (32/64)
- MIPS (32/64)

### Supported Compilers and Optimization Levels
- GCC (4.8, 5, 7, 9, 11)
- Clang (3.5, 5.0, 7, 9)
- Optimization Levels: O0, O1, O2, O3, Os

## Processing Your Own Binaries
If you want to apply CEBin to your own binaries, please refer to the scripts `pretrain/utils/dataset-prepare.py`

## Citation
If you find CEBin useful in your research, please consider citing our work:

```
@inproceedings{10.1145/3650212.3652117,
author = {Wang, Hao and Gao, Zeyu and Zhang, Chao and Sun, Mingyang and Zhou, Yuchen and Qiu, Han and Xiao, Xi},
title = {CEBin: A Cost-Effective Framework for Large-Scale Binary Code Similarity Detection},
year = {2024},
isbn = {9798400706127},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3650212.3652117},
doi = {10.1145/3650212.3652117},
abstract = {Binary code similarity detection (BCSD) is a fundamental technique for various applications. Many BCSD solutions have been proposed recently, which mostly are embedding-based, but have shown limited accuracy and efficiency especially when the volume of target binaries to search is large. To address this issue, we propose a cost-effective BCSD framework, CEBin, which fuses embedding-based and comparison-based approaches to significantly improve accuracy while minimizing overheads. Specifically, CEBin utilizes a refined embedding-based approach to extract features of target code, which efficiently narrows down the scope of candidate similar code and boosts performance. Then, it utilizes a comparison-based approach that performs a pairwise comparison on the candidates to capture more nuanced and complex relationships, which greatly improves the accuracy of similarity detection. By bridging the gap between embedding-based and comparison-based approaches, CEBin is able to provide an effective and efficient solution for detecting similar code (including vulnerable ones) in large-scale software ecosystems. Experimental results on three well-known datasets demonstrate the superiority of CEBin over existing state-of-the-art (SOTA) baselines. To further evaluate the usefulness of BCSD in real world, we construct a large-scale benchmark of vulnerability, offering the first precise evaluation scheme to assess BCSD methods for the 1-day vulnerability detection task. CEBin could identify the similar function from millions of candidate functions in just a few seconds and achieves an impressive recall rate of 85.46\% on this more practical but challenging task, which are several order of magnitudes faster and 4.07\texttimes{} better than the best SOTA baseline.},
booktitle = {Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis},
pages = {149–161},
numpages = {13},
keywords = {Binary Analysis, Deep Learning, Similarity Detection, Vulnerability Discovery},
location = {Vienna, Austria},
series = {ISSTA 2024}
}

@misc{wang2024cebincosteffectiveframeworklargescale,
      title={CEBin: A Cost-Effective Framework for Large-Scale Binary Code Similarity Detection}, 
      author={Hao Wang and Zeyu Gao and Chao Zhang and Mingyang Sun and Yuchen Zhou and Han Qiu and Xi Xiao},
      year={2024},
      eprint={2402.18818},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2402.18818}, 
}
```

Thank you for your interest in CEBin!