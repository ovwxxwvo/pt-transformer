import gradio as gr
import requests
import json


API_BASE_URL = "http://127.0.0.1:8000"

def train():
    url = f"{API_BASE_URL}/train"
    logs = []
    response = requests.post(url=url, stream=True, timeout=5)
    response.raise_for_status()

    for item in response.iter_lines(decode_unicode=True):
        if item:
            item = json.loads(item)
            if item["type"] == "init":
                desc = item["data"]["desc"]
                batch = item["data"]["batch"]
                logs.append(f"{desc} | batch:{batch}")
                yield gr.update(maximum=batch), "\n".join(logs)
            elif item["type"] == "update" and batch > 0:
                step = item["data"]["step"]
                loss = item["data"]["loss"]
                ids_pen = item["data"]["ids_pen"]
                logs.append(f"batch:{step}/{batch} | loss={loss:.4f}, ids_pen={ids_pen:.4f}")
                yield step, "\n".join(logs)
            elif item["type"] == "final":
                loss = item["data"]["loss"]
                ids_pen = item["data"]["ids_pen"]
                logs.append(f"{desc} | loss={loss:.4f} | ids_pen={ids_pen:.4f}")
                yield gr.update(maximum=batch), "\n".join(logs)

def eval():
    url = f"{API_BASE_URL}/eval"
    logs = []
    response = requests.post(url=url, stream=True, timeout=5)
    response.raise_for_status()

    for item in response.iter_lines(decode_unicode=True):
        if item:
            item = json.loads(item)
            if item["type"] == "init":
                desc = item["data"]["desc"]
                batch = item["data"]["batch"]
                logs.append(f"{desc} | batch:{batch}")
                yield gr.update(maximum=batch), "\n".join(logs)
            elif item["type"] == "update" and batch > 0:
                step = item["data"]["step"]
                loss = item["data"]["loss"]
                bleu = item["data"]["bleu"]
                logs.append(f"batch:{step}/{batch} | loss={loss:.4f}, bleu={bleu:.4f}")
                yield step, "\n".join(logs)
            elif item["type"] == "final":
                loss = item["data"]["loss"]
                bleu = item["data"]["bleu"]
                logs.append(f"{desc} | loss={loss:.4f} | bleu={bleu:.4f}")
                yield gr.update(maximum=batch), "\n".join(logs)

def infer(text):
    url = f"{API_BASE_URL}/infer"
    result = ""

    response = requests.post(
        url=url,
        json={"text": text},
        stream=True
        )
    response.raise_for_status()

    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            result += chunk.strip()
            yield result


with gr.Blocks() as app:
    gr.Markdown("# Transformer 翻译模型")

    with gr.Tab("训练"):
        eval_btn  = gr.Button("开始", variant="primary", size="lg")
        eval_pbar = gr.Slider(0, 100, value=0, label="进度", interactive=False)
        eval_log  = gr.Textbox(label="日志", interactive=False, lines=8, max_lines=16)
        eval_btn.click(fn=train, outputs=[eval_pbar, eval_log])

    with gr.Tab("验证"):
        eval_btn  = gr.Button("开始", variant="primary", size="lg")
        eval_pbar = gr.Slider(0, 100, value=0, label="进度", interactive=False)
        eval_log  = gr.Textbox(label="日志", interactive=False, lines=8, max_lines=16)
        eval_btn.click(fn=eval, outputs=[eval_pbar, eval_log])

    with gr.Tab("推理"):
        input_text  = gr.Textbox(label="输入文本", lines=3, autofocus=True)
        output_text = gr.Textbox(label="翻译结果", lines=3, interactive=False)
        gr.Button("开始翻译").click(fn=infer, inputs=input_text, outputs=output_text)
        # input_text.submit(fn=infer, inputs=input_text, outputs=output_text)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)


