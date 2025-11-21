from ..utils import config, variable


def main():
    print(f"-- Test Config & Variable --")
    print("=" * 40)

    print()
    Variable = variable.assign(config.load())
    v = Variable()

    print("\n🎉 Config & Variable test passed completely!")


if __name__ == "__main__":
    main()


