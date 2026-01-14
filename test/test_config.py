import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from common import load_config


def main():
    print(f"-- Test Config Module --")
    print("=" * 40)

    print()
    config = load_config()
    print(config)

    print("\n🎉 Config Module test passed completely!")


if __name__ == "__main__":
    main()


