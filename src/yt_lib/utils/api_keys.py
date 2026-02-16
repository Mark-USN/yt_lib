""" A simple API keys management class using python-dotenv."""
# import os
import dotenv
# import logging
from lib.utils.log_utils import get_logger # , log_tree

# from pathlib import Path

# -----------------------------
# Logging setup
# -----------------------------
logger = get_logger(__name__)


class api_vault(object):
    """A simple API class example."""

    def __init__(self,keys_file:str='.env'):
        try:
            self.keys_path = dotenv.find_dotenv(keys_file,
                                            raise_error_if_not_found=True)
            if dotenv.load_dotenv(self.keys_path):
                self.keys = dotenv.dotenv_values()

        except Exception as e:
            logger.error("Error loading keys from %s: %s", keys_file, e)
            raise e

    def get_value(self, key:str):
        value = self.keys[key]
        return value



if __name__ == "__main__":
    api = api_vault()
    key_value = api.get_value("GOOGLE_KEY")
    print(f"The value for Google_Key is: {key_value}")
