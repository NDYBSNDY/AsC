# Twice Embedding Space Optimized for Any-shot Classification Method with Wide Applicability in Several Industrial Fields

## Overview
> Industry-specific applications have raised significant requirements for few/zero-shot learning-based visual inspection, such as detecting rare-seen or unseen metal defects, surface damages, texture flaws, abnormal cell images, rare-seen disease diagnosis, and remote sensing classification. However, industry-specific data environments inevitably face inadequate training samples and insufficient label problems due to quite a few samples in the specific categories being hard to collect and label, such as defects, damages, rare-seen diseases, satellite images, etc. Moreover, existing industry-orient few-shot methods can not detect unseen samples (zero-shot), while general zero-shot methods perform poorly in industry-specific data environments. To this end, this work proposes a Generative and Contrastive Combined Support Sample Synthesis Model that can deal with both few-shot and zero-shot tasks in industry-specific data environments. First, a novel contrastive generator is proposed to use categories' labels to synthesize "fake" visual features for those without visual samples. Then, the synthesized visual features (for support samples) are fused with "real" visual features (for query samples) into a similarity graph to align the relationships between support samples and query samples. After, a class center optimization method is proposed to iteratively update the similarity matrix of the graph to obtain the classification probabilities for the query samples. Massive experiments on seven industry-specific datasets show the proposed method significantly surpasses all other peer methods. In the highlights, compared with state-of-the-art methods, our method gains an average of 8.29% improvements on few-shot tasks and achieves an average of 8.23% improvements on zero-shot tasks.

![generation_framework](./img/1.png)
---
## Prerequisites
To install all the dependency packages, please run:
```
pip install -r requirements.txt
```

---
## Data Preparation
1) We divide the seven different data sets as follows:
---
![generation_framework](./img/3.jpg)


2) Classification display of different datasets:
---
![generation_framework](./img/2.jpg)

3)Download link for processed data:
```
link：https://pan.baidu.com/s/1BiTVl4-TtYeo7quBjAYkAA 
password：6ypy						
```

## References
We adapt our dataloader classes from the following project:
https://github.com/successhaha/GTnet


