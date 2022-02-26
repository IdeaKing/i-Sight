# UDA Loss Function

![UDA Loss][https://1.bp.blogspot.com/-qu776D5N5Bc/XSZqJvILIdI/AAAAAAAAETA/Kzq1Qs4vsPQFmKpC_JNLc7-7kdM9GLphgCLcBGAs/s640/2019-07-10.jpg]

- There are two parts: Supervised Cross Entropy Loss & Unsupervised Consistency Loss
- UCL: Model is applied to both the unlabeled example and its augmented counterpart to produce two model predictions from which a consistency loss is computed
- Consistency is the distance between two prediction distributions. 
- https://ai.googleblog.com/2019/07/advancing-semi-supervised-learning-with.html
- https://github.com/lantgabor/Unsupervised-Data-Augmentation-PyTorch/blob/fef8c079445ccfa57823fcaa28a1d5ae5b71c052/UDA.py