import re, subprocess
from .paths import get_paths


def __get_changelog_raw() -> str:
    p = get_paths()
    CHANGELOG_PATH = p.ver_file
    if not CHANGELOG_PATH.exists():
        return {}
    with open(CHANGELOG_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    changelog_dict = {}
    version_re = re.compile(r"##\s+\[(.+?)\](.*?)(?=##\s+\[|\Z)", re.DOTALL | re.MULTILINE)
    category_re = re.compile(r"###\s+([A-Za-z]+)(.*?)(?=###\s+|\Z)", re.DOTALL | re.MULTILINE)
    item_re = re.compile(r"^\s*-\s+(.*?)$", re.MULTILINE)
    for version_match in version_re.finditer(raw):
        version = version_match.group(1).strip()
        version_content = version_match.group(2).strip()
        changelog_dict[version] = {}
        for category_match in category_re.finditer(version_content):
            category = category_match.group(1).strip()
            category_content = category_match.group(2).strip()
            items = [item.strip() for item in item_re.findall(category_content) if item.strip()]
            changelog_dict[version][category] = items
    return changelog_dict

def get_current_git_branch() -> str:
    branch = "master"
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True
            )
        branch = output.strip()
        return branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return branch

def get_version_str_stable() -> str:
    version = ""
    changelog = __get_changelog_raw()
    version_list = [v for v in changelog.keys() if v != "Unreleased"]
    version = version_list[0] if version_list else "v0.0.0"
    return version

def get_version_str_dev() -> str:
    version = ""
    changelog = __get_changelog_raw()
    stable_versions = [v for v in changelog.keys() if v != "Unreleased"]
    stable_ver = stable_versions[0] if stable_versions else "v0.0.0"
    unreleased = changelog.get("Unreleased", {})
    total_items = sum(len(items) for items in unreleased.values())

    ver_parts = stable_ver.lstrip("v").split(".")
    while len(ver_parts) < 3:
        ver_parts.append("0")
    if total_items > 0:
        ver_parts.append(str(total_items))
    else:
        if len(ver_parts) == 3:
            ver_parts.append("0")

    version = f"v{'.'.join(ver_parts)}"
    return version

def get_version_status() -> str:
    branch = get_current_git_branch()
    status = (branch == "master" and "stable" or "dev")
    return status

def get_version_str() -> str:
    status = get_version_status()
    version = (status == "stable" and get_version_str_stable() or get_version_str_dev())
    return version

def get_version_date() -> str:
    date = ""
    version_str = get_version_str()
    ver_parts = version_str.lstrip("v").split(".")
    while len(ver_parts) < 3:
        ver_parts.append("00")
    year, month, day = ver_parts[0], ver_parts[1], ver_parts[2]
    year = f"20{ver_parts[0]}" if len(ver_parts[0]) == 2 else ver_parts[0]
    date = f"{year}-{month}-{day}"
    return date

def get_changelog() -> dict:
    changelog = __get_changelog_raw()
    status = get_version_status()
    if status == "stable" and "Unreleased" in changelog:
        del changelog["Unreleased"]
    return changelog

def get_version_info() -> dict:
    return {
        "version": get_version_str(),
        "status":  get_version_status(),
        "date":    get_version_date(),
        "changelog": get_changelog()
        }


