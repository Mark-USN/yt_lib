""" A simple API keys management class using python-dotenv."""
# import os
import dotenv
from yt_lib.utils.log_utils import get_logger # , log_tree

# from pathlib import Path

# -----------------------------
# Logging setup
# -----------------------------
logger = get_logger(__name__)


class api_vault(object):
    """A simple API class example."""

    def __init__(self,keys_file:str='.env'):
        """ Initialize the API vault by loading keys from a .env file.
            Args:
                keys_file (str): The path to the .env file containing API keys.
            Raises:
                Exception: If there is an error loading the keys from the file.
        """
        try:
            self.keys_path = dotenv.find_dotenv(keys_file,
                                            raise_error_if_not_found=True)
            if dotenv.load_dotenv(self.keys_path):
                self.keys = dotenv.dotenv_values()

        except Exception as e:
            logger.error("Error loading keys from %s: %s", keys_file, e)
            raise e

    def get_value(self, key:str):
        """ Retrieve the value of a specific API key.
            Args:
                key (str): The name of the API key to retrieve.
            Returns:
                str: The value of the requested API key.
            Raises:
                KeyError: If the specified key is not found in the loaded keys.
        """
        value = self.keys[key]
        return value

if __name__ == "__main__":
    """ Example usage of the api_vault class to retrieve an API key value. """
    api = api_vault()
    key_value = api.get_value("GOOGLE_KEY")
    print(f"The value for Google_Key is: {key_value}")
