## PyTorch Transformer  

A lightweight learning demo of Transformer model implemented with PyTorch.  
Implements an end-to-end translation pipeline, dual deployment modes, metric visualizations, etc.  

----  
### 🔧 Features 特性  

- Provides Transformer model & related Neural Network Extension components.  
- Provides Transformer practical utility to streamline workflow.  
  ( data handling, model handling, metric tracking, loss penalizing, early stopping, etc )  
- Uses 100 English-Chinese text translation pairs to verify the full pipeline of training|evaluation|inference.  
- Tokenizes original text data by scripts to generate the tokenizer|vocab|tokenid.  
- Tests core modules (mask|model|version|config|variable|logger|database|cli).  
- Configures with a main config file & related sub-config files.  
- Parses command-line arguments (mode|epoch) with simplicity.  
- Visualizes LOSS & BLEU metrics via web browser, with metrics fetched from the database.  
- Runs directly locally, or is deployed with an API backend server to serve both terminal clients & webrowser clients.  

----  
### 🧩 Modules 模块  

- Core Module, Transformer model & related Neural Network Extension components & related practical utilities.  
- Common Module, project-level utilities functions and tools.  
- Script Module, text processing scripts, test scripts and multi-end interaction scripts.  

```  
.  
├── transformer/  
│   ├── nn_ext.py - Transformer Neural Network Extension components  
│   │   └── FeedForwardNetwork, MaskGenerator, ...  
│   ├── model.py  - Transformer Model, assembled & stacked by key layers  
│   │   └── EncodeLayer, DecodeLayer, InputLayer, OutputLayer, Transformer  
│   ├── monitor.py  - Transformer practical Utilities  
│   │   ├── DataHandler   ( reverse_vocab|batch_data|save_model_weight|... )  
│   │   └── ModelHandler  ( train_model|eval_model|infer_model )  
│   └── handler.py  - Transformer practical Utilities  
│       ├── MetricMeter   ( LossMeter|BleuMeter|... )  
│       ├── LossPenalizer ( IdsPenalizer|... )  
│       └── EarlyStopper  ( EarlyStopper(loss)|EarlyStopper(bleu)|... )  
│  
├── common/  
│   ├── paths.py    - common paths module, conf|log  
│   ├── version.py  - semantic versioning module, with YY.MM.DD.patch schema  
│   ├── config.py   - toml-based config module, loads config files to dict  
│   └── logger.py   - logging-based logger module, records model|main|server logs  
│  
├── utils/  
│   ├── paths.py    - common paths module, data|db  
│   ├── variable.py - toml-based variable module, converts config dict to usable variable  
│   ├── database.py - sqlite-based database module, stores loss|bleu metrics data  
│   └── cli.py      - argparse-based cli module, parses command-line arguments  
│  
├── data/    - data includes text|tokenizer|vocab|tokenid|model-weight|...  
├── scripts/ - scripts for processing & tokenizing text data  
├── test/    - scripts for testing mask|model|version|config|logger|variable|database|cli  
│  
├── config/     - sub-config files includes model|common|paths|utils  
├── config.toml - main configuration entry  
│  
├── main.py    - main entry with pipeline|train|eval|infer  
├── plotter.py - plotly-based visualizations, fetches metrics from database  
├── api.py     - fastapi-based RESTful API  
├── server.py  - uvicorn-based server  
├── termui.py  - textual-based terminal client  
├── webui.py   - gradio-based webrowser client  
│  
├── server.sh      - script to run the backend server, wrap `api.py` & `server.py`  
└── pt-transformer - script to run locally, wrap `python main.py`  
```  

----  
### 📦 Installation 安装  

#### Install manually 手动安装 (linux)  

Clone the repo to your project dir :  
```  
git clone https://github.com/ovwxxwvo/pt-transformer.git && cd pt-transformer  
```  

##### Option 1:  

- Run with Local Python(3.11+) Environment  
```  
# Create and activate a virtual environment  
python3 -m venv .venv && source .venv/bin/activate  
# Install dependencies  
pip install --upgrade pip && pip install -r requirements.txt  
# Start the service  
bash server.sh  
```  

- Run locally with `pt-transformer` or `python main.py`:  
```  
pt-transformer --total-stage-epoch 8  
```  
```  
pt-transformer --mode train --total-train-epoch 4  
```  
```  
pt-transformer --mode eval  --total-eval-epoch 4  
```  

- Visualize metrics with `python plotter.py`  
```  
python plotter.py  
```  

##### Option 2:  

- Run in Docker  
```  
# Build docker  
docker build -t pt-transformer:v1 .  
# Run docker  
docker run -p 8000:8000 -p 7860:7860 -v $(pwd)/data:/app/data pt-transformer:v1  
```  

- Open web browser and visit:  
```  
http://localhost:7860  
```  

----  
### 📝 Configuration 配置  

All configuration files are stored in the config/ folder in the project root directory.  
The main configuration entry is config.toml, which loads the following sub-configuration files.  
You can adjust parameters either by directly editing existing configuration files or by adding new ones.  

**model.toml** - Transformer Architecture  
```  
[transformer]  
  d_model      = 512     # Embedding dimension  
  d_ff         = 2048    # Feed-forward hidden dimension  
  n_heads      = 4       # Attention heads  
  enc_n_layers = 4       # Encoder layers  
  dec_n_layers = 4       # Decoder layers  
```  

**common.toml** - Epoch & Dataset  
```  
[epoch]  
  current_epoch     = 1  # Current round (resume training)  
  total_stage_epoch = 8  # Total pipeline rounds (train+eval+infer)  
  total_train_epoch = 1  # Single training round  
  total_eval_epoch  = 1  # Single evaluation round  
  total_infer_epoch = 1  # Single inference round  
[dataset]  
  batch_size    = 8      # Training batch size  
  max_seq_len   = 128    # Max sequence length (truncate/pad)  
```  

**utils.toml** – Optimizer & Scheduler  
```  
[optimizer]  
  lr           = "1e-5"  # Initial learning rate  
  weight_decay = "1e-6"  # L2 regularization coefficient  
[scheduler]  
  min_lr   = "1e-6"      # Minimum learning rate  
  mode     = "min"       # Adjust mode (metric minimization)  
  patience = 2           # Early stopping patience  
```  

----  
### 📜 [MIT](LICENSE) License 许可证  


