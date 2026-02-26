from src.core.config import Config
from src.app import AppConfig, BotApp

import logging
import datetime
import os

LOG_FILENAME = f"logs/bot_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    # filename=LOG_FILENAME,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    raw_cfg = Config.load("config/config.yaml").data

    # Boot-mode guard:
    # when started by systemd at boot (BOTFRIEND_BOOT_MODE=1), allow
    # configuration-driven disable without uninstalling/disabling the service.
    if os.getenv("BOTFRIEND_BOOT_MODE") == "1":
        autostart_cfg = raw_cfg.get("autostart", {}) if isinstance(raw_cfg, dict) else {}
        enabled = bool(autostart_cfg.get("enabled", True))
        if not enabled:
            logging.info(
                "Boot start is disabled in config (autostart.enabled=false). Exiting."
            )
            return

    cfg: AppConfig = AppConfig.from_dict(raw_cfg)

    logging.info("Starting bot with configuration: %s", cfg)
    app = BotApp(cfg)
    app.run_forever()


if __name__ == "__main__":
    main()
