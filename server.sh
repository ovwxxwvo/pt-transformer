#!/usr/bin/env bash


# uvicorn api:app --host 0.0.0.0 --port 8000

python ./server.py &
python ./webui.py  &


