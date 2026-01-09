## PyTorch Transformer  

A lightweight learning demo of Transformer model implemented with PyTorch.  
Implements an end-to-end translation pipeline, dual deployment modes, metric visualizations, etc.  

----  
### 🔧 Features 特性  
- Provides practical utility modules to streamline workflow.  
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
```  
.  
├── transformer/  
│   ├── nn_ext.py ( FeedForwardNetwork, MaskGenerator, ... )  
│   ├── model.py  ( EncodeLayer, DecodeLayer, InputLayer, OutputLayer, Transformer )  
│   └── utils.py  ( DataHandler, ModelHandler, MetricMeter, LossPenalizer, EarlyStopper, ... )  
│       ├── DataHandler   ( reverse_vocab|batch_data|save_model_weight|... )  
│       ├── ModelHandler  ( train_model|eval_model|infer_model )  
│       ├── MetricMeter   ( LossMeter|BleuMeter|... )  
│       ├── LossPenalizer ( IdsPenalizer|... )  
│       └── EarlyStopper  ( EarlyStopper(loss)|EarlyStopper(bleu)|... )  
│  
├── utils/  
│   ├── paths.py    ( common paths module, conf|log|db|data )  
│   ├── version.py  ( semantic versioning module, with YY.MM.DD.patch schema )  
│   ├── config.py   ( toml-based config module, loads config files to dict )  
│   ├── variable.py ( config to variable module, convert config dict to usable variable )  
│   ├── logger.py   ( logging-based logger module, records model|main|server logs )  
│   ├── database.py ( sqlite-based database module, stores loss|bleu metrics data )  
│   └── cli.py      ( argparse-based cli module, parses command-line arguments )  
│  
├── data/    ( text, tokenizer, vocab, tokenid, model-weight )  
├── scripts/ ( process_txt, tokenize_txt )  
├── test/    ( test mask|model|version|config|variable|logger|database|cli )  
│  
├── config/     ( model, common, paths, utils )  
├── config.toml ( main configuration entry )  
│  
├── main.py     ( main entry with pipeline|train|eval|infer )  
├── plotter.py  ( plotly-based visualizations, fetches metrics from database )  
├── api.py      ( fastapi-based RESTful API )  
├── server.py   ( uvicorn-based server )  
├── termui.py   ( textual-based terminal client )  
├── webui.py    ( gradio-based webrowser client )  
│  
└── server.sh   ( script to run the backend server )  
```  

----  
### 📦 Installation 安装  

#### Install manually 手动安装 (linux)  

Clone the repo to your project dir :  
```  
git clone https://github.com/ovwxxwvo/pt-transformer.git && cd pt-transformer  
```  

Option 1: Run with Local Python(3.11+) Environment  
```  
# Create and activate a virtual environment  
python3 -m venv .venv && source .venv/bin/activate  
# Install dependencies  
pip install --upgrade pip && pip install -r requirements.txt  
# Start the service  
bash server.sh  
```  

Run locally with `pt-transformer` or `python main.py`:  
```  
pt-transformer --total-stage-epoch 8  
```  
```  
pt-transformer --mode train --total-train-epoch 4  
```  
```  
pt-transformer --mode eval  --total-eval-epoch 4  
```  

Option 2: Run in Docker  
```  
# Build docker  
docker build -t pt-transformer:v1 .  
# Run docker  
docker run -p 8000:8000 -p 7860:7860 -v $(pwd)/data:/app/data pt-transformer:v1  
```  

Open web browser and visit:  
```  
http://localhost:7860  
```  

----  
### 📜 [MIT](LICENSE) License 许可证  


