import os, pathlib, toml


def load():
    # path = os.path.dirname(os.path.abspath(sys.argv[0]))
    path = pathlib.Path(__file__).parent.parent
    conf_dirt = os.path.join(path, "config")
    conf_file = os.path.join(path, "config.toml")
    # print(path)
    # print(conf_dirt)
    # print(conf_file)

    with open(conf_file, "r") as f:
        config = toml.load(f)
    # print(config)

    conf_files = config["include"]["conf_files"]
    # print(conf_files)

    for file in conf_files:
        path = os.path.join(conf_dirt, file)
        # print(path)
        with open(path, "r") as f:
            config.update(toml.load(f))
    # print(config)

    data_dirt = config["dirt"]["data_dirt"]
    # print(data_dir)

    print(f"conf_dirt : {conf_dirt}")
    print(f"data_dirt : {data_dirt}")
    print("-" * 40)
    return config


