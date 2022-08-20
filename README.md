# Driving Detection Server

## 简介

基于flask的服务器后端

---

## 使用

### 克隆本仓库
```
git clone https://github.com/cqbu/driving_detection_server.git
```

### 创建conda虚拟环境（可选）
```
conda create -n driving python=3.8 -y
conda activate driving
```

### 安装依赖
```
# 安装PyTorch（也可通过pip安装）：请确保cudatoolkit版本与系统cuda版本一致
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -r requirements.txt

# 编译CLRNet
cd CLRNet
python setup.py build develop
```
### 运行
```
python app.py
```
---
## 其他
- 本仓库使用的前端：[driving_detection_app](https://github.com/exhyy/driving_detection_app)
- [PyTorch安装](https://pytorch.org/get-started/locally/)
- [yolov5](https://github.com/ultralytics/yolov5)
- [CLRNet](https://github.com/Turoad/CLRNet)
