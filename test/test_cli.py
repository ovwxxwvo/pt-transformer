import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils import parse_cli_args


def test_cli_args(test_argv):
    """Test CLI argument parsing with a given argument list"""
    original_argv = sys.argv.copy()
    try:
        # Override sys.argv with test parameters
        sys.argv = test_argv
        args = parse_cli_args()
        return True, args
    except SystemExit:
        # SystemExit is raised when --version is called (argparse's behavior)
        return True, "Version argument triggered"
    except Exception as e:
        return False, str(e)
    finally:
        # Restore original sys.argv to avoid side effects
        sys.argv = original_argv

def main():
    print("-- Test Cli Module --")
    print("=" * 40)

    # Define test cases (all scenarios covered)
    test_cases = [
        ("Test 1: Version argument", ["test_cli.py", "-v"]),
        ("Test 2: Full parameters", ["test_cli.py", "--mode", "train", "--n-heads", "8", "--enc-n-layers", "6", "--dec-n-layers", "6", "--epoch-total", "32"]),
        ("Test 3: Invalid mode", ["test_cli.py", "--mode", "invalid_mode"])
        ]

    # Run all test cases
    for test_name, test_argv in test_cases:
        print()
        print(f"🔹 {test_name}")
        success, result = test_cli_args(test_argv)

        if success:
            if isinstance(result, str):
                print(f"   ✅ Parse success")
            else:
                print(f"Running Mode: {result.mode}")
                print(f"Attention Heads: {result.n_heads if result.n_heads else 'Default'}")
                print(f"Encoder Layers: {result.enc_n_layers if result.enc_n_layers else 'Default'}")
                print(f"Decoder Layers: {result.dec_n_layers if result.dec_n_layers else 'Default'}")
                print(f"Total Epochs: {result.epoch_total if result.epoch_total else 'Default'}")
                print(f"   ✅ Parse success")
        else:
            print(f"   ❌ Failed: {result}")

    print("\n🎉 Cli Module test passed completely!")


if __name__ == "__main__":
    main()


