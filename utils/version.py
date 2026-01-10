# -------------------------- Core: Version Identification (Only modify here when iterating) --------------------------
# Version format: YYYY.MM.DD.patch (aligns with semantic versioning logic)
# - YY.MM.DD: Main/minor version (marks core iteration date)
# - patch: Revision version (increments for bug fixes/minor optimizations)
# Version Status: stable/beta/dev (for environment distinction)
__version__ = "25.12.01.0"
__status__ = "stable"

# -------------------------- Auxiliary: Version Changelog (Structured for traceability) --------------------------
# Key: Version number; Value: List of changes (classified by "Add/Opt/Fix" for clarity)
version_changelog = {
    "25.12.01.0": [
        "【Add】nn_ext.py - Transformer Neural Network Extension components ( FeedForwardNetwork, MaskGenerator, ... )",
        "【Add】model.py  - Transformer Model, assembled & stacked by key layers ( EncodeLayer, DecodeLayer, InputLayer, OutputLayer, Transformer )",
        "【Add】utils.py  - Transformer practical Utilities ( DataHandler, ModelHandler, MetricMeter, LossPenalizer, EarlyStopper, ... )",
        "【Add】paths.py    - common paths module, conf|log|db|data                             ",
        "【Add】version.py  - semantic versioning module, with YY.MM.DD.patch schema            ",
        "【Add】config.py   - toml-based config module, loads config files to dict              ",
        "【Add】variable.py - config to variable module, converts config dict to usable variable ",
        "【Add】logger.py   - logging-based logger module, records model|main|server logs       ",
        "【Add】database.py - sqlite-based database module, stores loss|bleu metrics data       ",
        "【Add】cli.py      - argparse-based cli module, parses command-line arguments          ",
        "【Add】test/ - scripts for testing mask|model|version|config|variable|logger|database|cli",
        "【Add】main.py    - main entry with pipeline|train|eval|infer                 ",
        "【Add】plotter.py - plotly-based visualizations, fetches metrics from database",
        "【Add】api.py     - fastapi-based RESTful API                                 ",
        "【Add】server.py  - uvicorn-based server                                      ",
        "【Add】termui.py  - textual-based terminal client                             ",
        "【Add】webui.py   - gradio-based webrowser client                             ",
        "【Add】server.sh      - script to run the backend server, wrap `api.py` & `server.py`",
        "【Add】pt-transformer - script to run locally, wrap `python main.py`                 ",
        "【Add】 ",
        "【Add】 ",
        ],
    }

# -------------------------- Utility Functions: Expose Version Information --------------------------

def get_version_str() -> str:
    """
    Returns only the version string for simple use cases (e.g., API response, logger injection)
    :return: Pure version string (e.g., "25.12.01.0")
    """
    return __version__

def get_version_info() -> dict:
    """
    Returns structured version information for logger, database, UI, etc.
    :return: Dict containing version, update date, changelog, and status
    """
    version_parts = __version__.split(".")
    update_date = f"20{version_parts[0]}-{version_parts[1]}-{version_parts[2]}"

    return {
        "version":     __version__,
        "update_date": update_date,
        "changelog":   version_changelog.get(__version__, []),
        "status":      __status__,
        }


