import os, pathlib, logging
from logging.handlers import TimedRotatingFileHandler


path = pathlib.Path(__file__).parent.parent
log_dirt = os.path.join(path, "logs")
os.makedirs(log_dirt, exist_ok=True)

class VersionFilter(logging.Filter):
    def filter(self, record):
        record.version = "25.12.01"
        return True

# 初始化全局单例logger
def init_logger():
    logger = logging.getLogger("PT_TRANSFORMER")
    logger.setLevel(logging.DEBUG)  # 全局日志级别（最低级别，确保所有日志可捕获）
    logger.addFilter(VersionFilter())  # 注入版本号到日志
    logger.propagate = False  # 避免重复输出

    # 日志格式：时间 | 版本 | 模块:行号 | 级别 | 内容（清晰可追溯）
    formatter = logging.Formatter(
        "%(asctime)s | %(version)s | %(module)s:%(lineno)d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 控制台Handler：输出INFO及以上级别（开发调试用，过滤冗余调试日志）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件Handler：按日期切割，输出DEBUG及以上级别（持久化存档，保留7天）
    log_file_prefix = os.path.join(log_path, "pt_transformer")  # 日志文件前缀
    file_handler = TimedRotatingFileHandler(
        filename=log_file_prefix,
        when="D",  # 按天切割
        interval=1,  # 每天1个文件
        backupCount=7,  # 保留7天历史日志
        encoding="utf-8",  # 避免中文乱码
        suffix="%Y-%m-%d.log"  # 日期前置到.log前，最终格式：pt_transformer_2025-12-01.log
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 全局唯一logger实例，全项目直接导入使用
logger = init_logger()

# 测试代码（运行logger.py可验证，后续可删除）
if __name__ == "__main__":
    logger.debug("调试日志：仅写入文件，控制台不显示")
    logger.info("信息日志：控制台+文件均显示")
    logger.warning("警告日志：提示潜在风险")
    try:
        1 / 0
    except Exception as e:
        logger.error(f"错误日志：{str(e)}", exc_info=True)  # 带异常堆栈，便于排查
    logger.critical("致命日志：项目无法继续运行")


