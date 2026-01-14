# Changelog  

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/):  
    - **Added** for new features.  
    - **Changed** for changes in existing functionality.  
    - **Deprecated** for soon-to-be removed features.  
    - **Removed** for now removed features.  
    - **Fixed** for any bug fixes.  
    - **Security** in case of vulnerabilities.  

## [Unreleased]  

## [v26.01.01]  

  ### Added  
    - nn_ext.py - Transformer Neural Network Extension components  
      ( FeedForwardNetwork, MaskGenerator, ... )  
    - model.py  - Transformer Model, assembled & stacked by key layers  
      ( EncodeLayer, DecodeLayer, InputLayer, OutputLayer, Transformer )  
    - utils.py  - Transformer practical Utilities  
      ( DataHandler, ModelHandler, MetricMeter, LossPenalizer, EarlyStopper, ... )  
    - paths.py    - common paths module, data|db  
    - variable.py - toml-based variable module, converts config dict to usable variable  
    - database.py - sqlite-based database module, stores loss|bleu metrics data  
    - cli.py      - argparse-based cli module, parses command-line arguments  
    - paths.py    - common paths module, conf|log  
    - version.py  - semantic versioning module, with YY.MM.DD.patch schema  
    - config.py   - toml-based config module, loads config files to dict  
    - logger.py   - logging-based logger module, records model|main|server logs  
    - test/ - scripts for testing mask|model|version|config|variable|logger|database|cli  
    - main.py    - main entry with pipeline|train|eval|infer  
    - plotter.py - plotly-based visualizations, fetches metrics from database  
    - api.py     - fastapi-based RESTful API  
    - server.py  - uvicorn-based server  
    - termui.py  - textual-based terminal client  
    - webui.py   - gradio-based webrowser client  
    - server.sh      - script to run the backend server, wrap `api.py` & `server.py`  
    - pt-transformer - script to run locally, wrap `python main.py`  

  ### Fixed  
    - test  
    - test  


