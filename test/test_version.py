import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.version import (
    get_current_git_branch,
    get_version_status,
    get_version_str_stable,
    get_version_str_dev,
    get_version_str,
    get_version_date,
    get_changelog,
    get_version_info
    )


def main():
    print(f"-- Test Version Module --")
    print("=" * 40)

    print()
    print(f"{'当前分支'}:     {get_current_git_branch()}")
    print(f"{'版本状态'}:     {get_version_status()}")
    print(f"{'稳定版版本号'}: {get_version_str_stable()}")
    print(f"{'开发版版本号'}: {get_version_str_dev()}")
    print(f"{'当前版本号'}:   {get_version_str()}")
    print(f"{'版本日期'}:     {get_version_date()}")

    print(f"\n{'变更日志(字典)'}: ")
    changelog = get_changelog()
    for ver, detail in changelog.items():
        print(f"{ver}: {detail}")

    print(f"\n{'完整版本信息'}: ")
    version_info = get_version_info()
    for key, value in version_info.items():
        print(f"{key}: {value}")

    print("\n🎉 Version Module test passed completely!")


if __name__ == "__main__":
    main()


