import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import load_config, create_variable


def main():
    print(f"-- Test Config & Variable --")
    print("=" * 40)

    print()
    Variable = create_variable(load_config())
    v = Variable()

    print("\n🎉 Config & Variable test passed completely!")


if __name__ == "__main__":
    main()


