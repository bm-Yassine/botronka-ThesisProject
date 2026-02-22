from src.core.config import Config
from src.app import AppConfig, BotApp

import logging
import datetime

LOG_FILENAME = f"logs/bot_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    # filename=LOG_FILENAME,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    cfg: AppConfig = AppConfig.from_dict(Config.load("config/config.yaml").data)

    logging.info("Starting bot with configuration: %s", cfg)
    app = BotApp(cfg)
    app.run_forever()


if __name__ == "__main__":
    main()
