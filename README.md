## 项目介绍
- RAG前的预处理，将大多数文件转成Markdown

### 环境配置
#### MinerU环境配置
1.  conda create -n MinerU python=3.10
2.  conda activate MinerU
3.  pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple
4.  下载模型权重文件
    1.  pip install huggingface_hub
    2.  wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models_hf.py -O download_models_hf.py
        1.  如果没有wget，换成如下命令
        2.   certutil -urlcache -split -f "https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py" download_models.py
        
    3.  python download_models_hf.py
- 参考
  - https://mineru.readthedocs.io/zh-cn/latest/user_guide/install.html
#### 为了解析除pdf外的各种类型文档，我们需要下载LibreOffice，Windows操作如下
1. 首先要安装LibreOffice 
   1. https://zh-cn.libreoffice.org/download/libreoffice/ 
2. 然后将其中的program文件路径加入到环境变量接口；
#### 安装Ollama，使用本地视觉模型
  - pip install ollama
    - my ollama version is 0.5.7，你可以检查一下
  - 下载ollama客户端 https://ollama.com/
  - ollama run minicpm-v
    
#### 快速开始
- git clone git@github.com:wcool1/PreRAG.git
- cd PreRAG
- cd demo
- 删除ouput文件夹
- python demo.py
  - 可以在demo.py中修改文件路径
### 参考与致谢
- [MinerU]( https://github.com/opendatalab/MinerU)
- [ollama](https://github.com/ollama/ollama-python)