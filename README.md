<h1 align="center">
DSBench: A Comprehensive Benchmark for Evaluating External and In-Cabin Risks
</h1>

<p align='center'>
<img align="center" src='images/intro.png' width='100%'> </img>
</p>

<h5 align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-9BDFDF)](https://github.com/linglingxiansen/SpatialSky/blob/main/LICENSE) 
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoint-FBD49F.svg)](https://huggingface.co/mengxxianhhui/DSVLM)
[![arXiv](https://img.shields.io/badge/Arxiv-2511.13269-E69191.svg?logo=arXiv)](https://arxiv.org/abs/2511.14592) 


# 🚀 Getting Start
## ⚙️ Installation
```
git clone https://github.com/mengxh20/DSBench.git
conda create -n dsbench python=3.12 -y
conda activate dsbench
pip install torch torchvision torchaudio 
pip install openai transformers
```
## 🚗 Benchmark
Our benchmark can be available at [here](https://pan.baidu.com/s/1ER6buToR4rrEG5e4pHxdrQ?pwd=zivt). 

## ⬇️ Download Ckpt

Download our DSVLM model from [![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoint-FBD49F.svg)](https://huggingface.co/mengxxianhhui/DSVLM).


## 🔍 Inference
Our inference code can be found in the "inference" folder.

For commercial models run:
```
python inference/inference_close.py
``` 

For open source models run:
```
python inference/inference.py
``` 

If using a local deployment approach, run:
```
python inference/inference_close.py
``` 

## 🤖 Evaluation
Evaluating the results:
```
bash evaluation/metrics.sh
```



## Citation
If this work is helpful, please kindly cite as:
```
@article{meng2025your,
  title={Is Your VLM for Autonomous Driving Safety-Ready? A Comprehensive Benchmark for Evaluating External and In-Cabin Risks},
  author={Meng, Xianhui and Zhang, Yuchen and Huang, Zhijian and Lu, Zheng and Ji, Ziling and Yin, Yaoyao and Zhang, Hongyuan and Jiang, Guangfeng and Lin, Yandan and Chen, Long and others},
  journal={arXiv preprint arXiv:2511.14592},
  year={2025}
}
```
