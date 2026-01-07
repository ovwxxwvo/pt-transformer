import os, pathlib, webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots as ms
from utils import get_metric_db


proj_root = pathlib.Path(__file__).parent
html_file = os.path.join(proj_root, "metric.html")

class MetricPlotter:
    def __init__(self):
        self.db = get_metric_db()
        self.train_data = []
        self.eval_data = []
        self._load_metrics()
        self._parse_metrics()

    def _load_metrics(self):
        self.train_data = self.db.query_metrics("train")
        self.eval_data = self.db.query_metrics("eval")

    def _parse_metrics(self):
        self.train_epochs = [row[0] for row in self.train_data]
        self.train_loss = [row[7] for row in self.train_data]
        self.train_bleu = [row[8] for row in self.train_data]

        self.eval_epochs = [row[0] for row in self.eval_data]
        self.eval_loss = [row[7] for row in self.eval_data]
        self.eval_bleu = [row[8] for row in self.eval_data]

    def create_plots(self):
        # Create subplots with 2 independent legends, vertical spacing 0.15
        fig = ms(
            rows=2, cols=1,
            subplot_titles=("LOSS", "BLEU"),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )

        # Add LOSS traces, bind to legend 1
        fig.add_trace(
            go.Scatter(
                x=self.train_epochs,
                y=self.train_loss,
                name="Train LOSS",
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6, color="#1f77b4"),
                hovertemplate="Global Epoch: %{x}<br>LOSS: %{y:.4f}<extra></extra>",
                legendgroup="loss",
                legend="legend1"
            ),
            row=1, col=1
        )

        if self.eval_loss:
            fig.add_trace(
                go.Scatter(
                    x=self.eval_epochs,
                    y=self.eval_loss,
                    name="Eval LOSS",
                    mode="lines+markers",
                    line=dict(color="#1f77b4", width=2, dash="dash"),
                    marker=dict(size=6, color="#1f77b4", symbol="circle-open"),
                    hovertemplate="Global Epoch: %{x}<br>LOSS: %{y:.4f}<extra></extra>",
                    legendgroup="loss",
                    legend="legend1"
                ),
                row=1, col=1
            )

        # Add BLEU traces, bind to legend 2
        if self.train_bleu and any(b is not None for b in self.train_bleu):
            fig.add_trace(
                go.Scatter(
                    x=self.train_epochs,
                    y=self.train_bleu,
                    name="Train BLEU",
                    mode="lines+markers",
                    line=dict(color="#ff7f0e", width=2),
                    marker=dict(size=6, color="#ff7f0e"),
                    hovertemplate="Global Epoch: %{x}<br>BLEU: %{y:.4f}<extra></extra>",
                    legendgroup="bleu",
                    legend="legend2"
                ),
                row=2, col=1
            )

        if self.eval_bleu and any(b is not None for b in self.eval_bleu):
            fig.add_trace(
                go.Scatter(
                    x=self.eval_epochs,
                    y=self.eval_bleu,
                    name="Eval BLEU",
                    mode="lines+markers",
                    line=dict(color="#ff7f0e", width=2, dash="dash"),
                    marker=dict(size=6, color="#ff7f0e", symbol="circle-open"),
                    hovertemplate="Global Epoch: %{x}<br>BLEU: %{y:.4f}<extra></extra>",
                    legendgroup="bleu",
                    legend="legend2"
                ),
                row=2, col=1
            )

        # Layout config: separate horizontal legends for each subplot
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Transformer Metrics",
            title_x=0.5,
            hovermode="x unified",
            # LOSS legend: horizontal, top-right of LOSS subplot
            legend1=dict(
                orientation="h",
                yanchor="top",
                xanchor="right",
                y=0.95,
                x=0.98,
                tracegroupgap=5
            ),
            # BLEU legend: horizontal, top-right of BLEU subplot
            legend2=dict(
                orientation="h",
                yanchor="top",
                xanchor="right",
                y=0.40,
                x=0.98,
                tracegroupgap=5
            )
        )

        # Axis config
        fig.update_xaxes(title_text="Global Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Global Epoch", row=2, col=1)
        fig.update_yaxes(title_text="LOSS Value", row=1, col=1, tickformat=".4f")
        fig.update_yaxes(title_text="BLEU Value", row=2, col=1, tickformat=".4f")

        fig.write_html(html_file)
        print(f"✅ Metric visualization saved to: {html_file}")
        self._open_browser()

    def _open_browser(self):
        if os.path.exists(html_file):
            abs_path = os.path.abspath(html_file)
            webbrowser.open(f"file://{abs_path}")
            print(f"🌐 Opened metric.html in default browser!")
        else:
            print("❌ Failed to find metric.html file!")

def main():
    try:
        plotter = MetricPlotter()
        plotter.create_plots()
    except Exception as e:
        print(f"❌ Error generating plots: {str(e)}")


if __name__ == "__main__":
    main()


