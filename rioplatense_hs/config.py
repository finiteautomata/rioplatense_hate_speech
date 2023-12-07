import os
import configparser

config = configparser.ConfigParser()

# Get parent directory
current_path = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(current_path, "..", "config")

config.read(os.path.join(config_path, "config.ini"))
