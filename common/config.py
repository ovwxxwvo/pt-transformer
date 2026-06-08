import os, toml
from .paths import get_paths


def load_config():
    # path = os.path.dirname(os.path.abspath(sys.argv[0]))
    p = get_paths()
    proj_root = p.proj_root
    conf_file = p.conf_file
    # print(proj_root)
    # print(conf_file)

    with open(conf_file, "r") as f:
        config = toml.load(f)
    # print(config)

    conf_files = config["include"]["conf_files"]
    # print(conf_files)

    for file in conf_files:
        path = os.path.join(proj_root, file)
        # print(path)
        with open(path, "r") as f:
            config.update(toml.load(f))
    # print(config)

    # print("-" * 40)
    return config


config = load_config()


