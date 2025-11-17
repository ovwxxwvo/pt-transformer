## PyTorch Transformer  

Transformer model implemented with PyTorch.  

----  

### 🧩 Modules 模块  

Basic modules (`./transformer/nn_ext.py`) :  
  - FeedForwardNetwork, MaskGenerator, ...  

Core modules (`./transformer/model.py`):  
  - EncodeLayer, DecodeLayer, InputLayer, OutputLayer, Transformer.  

Util modules (`./transformer/utils.py`):  
  - DataHandler, ModelHandler, MetricMeter, LossPenalizer, EarlyStopper, ...  

----  

### 🔧 Function 功能  

DataHandler  
  - **`reverse_vocab`|`batch_data`|`save_log`|`save_model_weight`| ...**  

ModeHandler  
  - **`train_model`|`eval_model`|`infer_model`**  

MetricMeter  
  - **`LossMeter`|`BleuMeter`| ...**  

LossPenalizer  
  - **`IdsPenalizer`| ...**  

EarlyStopper  
  - **`EarlyStopper(loss)`|`EarlyStopper(bleu)`| ...**  

----  

### 📦 Installation 安装  

#### Install manually 手动安装 (linux)  

Clone the repo to your project dir :  
```  
git clone https://github.com/ovwxxwvo/pt-transformer.git  
```  
Keep only `Transformer` core dir in project :  
```  
cd pt-transformer && mv transformer ../ && cd ../ && rm -rf pt-transformer  
```  

----  

### 📜 [MIT](LICENSE) License 许可证  


