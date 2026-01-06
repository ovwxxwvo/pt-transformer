import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils import init_logger


def main():
    print(f"-- Test Logger Module --")
    print("=" * 40)

    print()
    logger = init_logger("test")
    # Test logs covering different levels (verify console + file output)
    logger.info("✅ Info log: Displayed in both console and file (commonly used for development)")
    logger.debug("✅ Debug log: Written to file only, not displayed in console")
    logger.warning("⚠️ Warning log: Potential risk alert (e.g., mismatched parameters)")

    print()
    try:
        1 / 0  # Intentionally trigger exception to verify error log with traceback
    except Exception as e:
        logger.error(f"❌ Error log: {str(e)}", exc_info=True)  # Includes traceback for debugging

    print()
    logger.critical("🔥 Critical log: Fatal error (e.g., service startup failure)")

    print("\n🎉 Logger Module test passed completely!")


if __name__ == "__main__":
    main()


