## PyTorch Transformer  

Transformer model implemented with PyTorch.  

### 🧩 Modules 模块  
----  
```  
.  
├── transformer/  
│   ├── nn_ext.py ( FeedForwardNetwork, MaskGenerator, ... )  
│   ├── model.py  ( EncodeLayer, DecodeLayer, InputLayer, OutputLayer, Transformer )  
│   └── utils.py  ( DataHandler, ModelHandler, MetricMeter, LossPenalizer, EarlyStopper, ... )  
│       ├── DataHandler   (reverse_vocab|batch_data|save_model_weight|...)  
│       ├── ModelHandler  (train_model|eval_model|infer_model)  
│       ├── MetricMeter   (LossMeter|BleuMeter|...)  
│       ├── LossPenalizer (IdsPenalizer|...)  
│       └── EarlyStopper  (EarlyStopper(loss)|EarlyStopper(bleu)|...)  
│  
├── utils/  
│   ├── version.py  ( semantic versioning module, with YY.MM.DD.patch schema )  
│   ├── config.py   ( toml-based config module, loads config files to dict )  
│   ├── variable.py ( config to variable module, convert config dict to usable variable )  
│   ├── logger.py   ( logging-based logger module, records model|main|server logs )  
│   ├── database.py ( sqlite-based database module, stores loss|bleu metrics data )  
│   └── cli.py      ( argparse-based cli module, parses command-line arguments )  
│  
├── data/    ( text, tokenizer, vocab, tokenid, model-weight )  
├── scripts/ ( process_txt, tokenize_txt )  
├── config/  ( model, common, paths, utils )  
├── test/    ( test mask|model|version|config|variable|logger|database|cli )  
│  
├── config.toml ( main configuration entry )  
├── main.py     ( main entry with pipeline|train|eval|infer )  
├── plotter.py  ( plotly-based visualizations, fetches metrics from database )  
├── api.py      ( fastapi-based RESTful API )  
├── server.py   ( uvicorn-based server )  
├── termui.py   ( textual-based terminal client )  
├── webui.py    ( gradio-based webrowser client )  
└── server.sh   ( script for run server )  
```  

----  

### 📦 Installation 安装  

#### Install manually 手动安装 (linux)  

Clone the repo to your project dir :  
```  
git clone https://github.com/ovwxxwvo/pt-transformer.git  
```  

----  
### 📜 [MIT](LICENSE) License 许可证  


