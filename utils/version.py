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
        "【Add】Project infrastructure setup (Transformer core model, data preprocessing scripts)",
        "【Add】Logger module (dual-output + daily rotation + version binding)",
        "【Add】Multi-end interaction support (WebUI, TermUI, API interface)",
        "【Opt】Data preprocessing logic (improved efficiency of CSV tokenization)"
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


