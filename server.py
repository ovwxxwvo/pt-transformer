from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn, json
import model


v, m, mh, dh = model.init()

class Request(BaseModel):
    text: str

app = FastAPI(title="Transformer翻译")

@app.post("/workflow")
def workflow():
    model.workflow(v, m, mh, dh)

@app.post("/train")
def train():
    generator = mh.train_model(v.databatchs_train, v.optimizer, v.metricmeter, v.penalizer)
    response = StreamingResponse(
        ((json.dumps(data)+"\n").encode("utf-8") for data in generator),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return response

@app.post("/eval")
def eval():
    generator = mh.eval_model(v.databatchs_eval, v.scheduler, v.metricmeter)
    response = StreamingResponse(
        ((json.dumps(data)+"\n").encode("utf-8") for data in generator),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return response

@app.post("/infer")
def infer(request:Request):
    text = request.text
    text_gen = mh.infer_model(text, v.seq_len,
        v.pad_id, v.unk_id, v.sos_id, v.eos_id,
        v.tokenizer_src,
        v.tokenizer_tgt,
        )
    response = StreamingResponse(
        (word + "\n" for word in text_gen),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return response


if __name__ == "__main__":
    uvicorn.run(
        app="server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
        )


