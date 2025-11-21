from textual.app import App, ComposeResult
from textual.widgets import TabbedContent, TabPane, Input, TextArea, Button, ProgressBar
from textual.containers import Vertical
import requests, json
import threading


API_BASE_URL = "http://127.0.0.1:8000"

ID_TRAIN_PROGRESS= "train_progress"
ID_TRAIN_OUTPUT  = "train_output"
ID_TRAIN_BOTTON  = "train_botton"
ID_EVAL_PROGRESS = "eval_progress"
ID_EVAL_OUTPUT   = "eval_output"
ID_EVAL_BOTTON   = "eval_botton"
ID_INFER_INPUT   = "infer_input"
ID_INFER_OUTPUT  = "infer_output"
ID_INFER_BOTTON  = "infer_botton"

class Application(App):
    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("train"):
                with Vertical():
                    yield ProgressBar(id=ID_TRAIN_PROGRESS)
                    yield TextArea(id=ID_TRAIN_OUTPUT, placeholder="Log ...", read_only=True)
                    yield Button(id=ID_TRAIN_BOTTON, label="Start Train")
            with TabPane("Eval"):
                with Vertical():
                    yield ProgressBar(id=ID_EVAL_PROGRESS)
                    yield TextArea(id=ID_EVAL_OUTPUT, placeholder="Log ...", read_only=True)
                    yield Button(id=ID_EVAL_BOTTON, label="Start Eval")
            with TabPane("Infer"):
                with Vertical():
                    yield Input(id=ID_INFER_INPUT, placeholder="Please input text ...")
                    yield TextArea(id=ID_INFER_OUTPUT, placeholder="Wait for infer ...", read_only=True)
                    yield Button(id=ID_INFER_BOTTON, label="Start Infer")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "train_botton":
                threading.Thread(target=self.train,  daemon=True).start()
            case "eval_botton":
                threading.Thread(target=self.eval,  daemon=True).start()
            case "infer_botton":
                threading.Thread(target=self.infer, daemon=True).start()


    def train(self) -> None:
        url = f"{API_BASE_URL}/train"
        progress_comp = self.query_one(f"#{ID_TRAIN_PROGRESS}")
        output_comp = self.query_one(f"#{ID_TRAIN_OUTPUT}")

        response = requests.post(url=url, stream=True, timeout=5)
        response.raise_for_status()

        for item in response.iter_lines(decode_unicode=True):
            if item:
                item = json.loads(item)
                if item["type"] == "init":
                    desc = item["data"]["desc"]
                    batch = item["data"]["batch"]
                    log = f"{desc} | batch:{batch}"
                    output_comp.text = f"{log}\n"
                    progress_comp.update(total=batch, progress=0)
                elif item["type"] == "update" and batch > 0:
                    step = item["data"]["step"]
                    loss = item["data"]["loss"]
                    ids_pen = item["data"]["ids_pen"]
                    log = f"batch:{step}/{batch} | loss={loss:.4f}, ids_pen={ids_pen:.4f}"
                    output_comp.text += f"{log}\n"
                    output_comp.move_cursor((step, 0))
                    progress_comp.advance(1)
                elif item["type"] == "final":
                    loss = item["data"]["loss"]
                    ids_pen = item["data"]["ids_pen"]
                    log = f"{desc} | loss={loss:.4f}, ids_pen={ids_pen:.4f}"
                    output_comp.text += f"{log}\n"

    def eval(self) -> None:
        url = f"{API_BASE_URL}/eval"
        progress_comp = self.query_one(f"#{ID_EVAL_PROGRESS}")
        output_comp = self.query_one(f"#{ID_EVAL_OUTPUT}")

        response = requests.post(url=url, stream=True, timeout=5)
        response.raise_for_status()

        for item in response.iter_lines(decode_unicode=True):
            if item:
                item = json.loads(item)
                if item["type"] == "init":
                    desc = item["data"]["desc"]
                    batch = item["data"]["batch"]
                    log = f"{desc} | batch:{batch}"
                    output_comp.text = f"{log}\n"
                    progress_comp.update(total=batch, progress=0)
                elif item["type"] == "update" and batch > 0:
                    step = item["data"]["step"]
                    loss = item["data"]["loss"]
                    bleu = item["data"]["bleu"]
                    log = f"batch:{step}/{batch} | loss={loss:.4f}, bleu={bleu:.4f}"
                    output_comp.text += f"{log}\n"
                    output_comp.move_cursor((step, 0))
                    progress_comp.advance(1)
                elif item["type"] == "final":
                    loss = item["data"]["loss"]
                    bleu = item["data"]["bleu"]
                    log = f"{desc} | loss={loss:.4f}, bleu={bleu:.4f}"
                    output_comp.text += f"{log}\n"

    def infer(self) -> None:
        url = f"{API_BASE_URL}/infer"
        input_comp = self.query_one(f"#{ID_INFER_INPUT}")
        output_comp = self.query_one(f"#{ID_INFER_OUTPUT}")

        response = requests.post(
            url=url,
            json={"text": input_comp.value.strip()},
            stream=True,
            )
        response.raise_for_status()

        output_comp.text = ""
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                output_comp.text += chunk.strip()


if __name__ == "__main__":
    Application().run()


