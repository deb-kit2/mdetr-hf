import logging


if __name__ == "__main__" :

    logging.basicConfig(
        level = logging.DEBUG,
        handlers = [
            # logging.FileHandler(f"{args.output_dir}/{args.experiment_name}.log"),
            logging.StreamHandler()
        ],
        format = "%(asctime)s %(levelname)s: \t%(message)s"
    )

    logging.error("You have made an error!")

    print("Hello!")