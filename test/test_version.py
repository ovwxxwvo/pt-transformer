import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils import get_version_str, get_version_info


def main():
    print(f"-- Test Version Module --")
    print("=" * 40)

    print()
    print("Version: " + get_version_str())
    print(get_version_info())

    print("\n🎉 Version Module test passed completely!")


if __name__ == "__main__":
    main()


