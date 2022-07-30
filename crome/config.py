import configparser
import os

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

ENVIRONMENT = 'production'
CONFIG_PATH = os.path.join(PROJECT_PATH, 'config.prod.ini')

if not os.path.exists(CONFIG_PATH):
    ENVIRONMENT = 'development'
    CONFIG_PATH = os.path.join(PROJECT_PATH, 'config.dev.ini')

if not os.path.exists(CONFIG_PATH):
    ENVIRONMENT = 'default'
    CONFIG_PATH = os.path.join(PROJECT_PATH, 'config.ini')

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

config['DEFAULT']['project_path'] = PROJECT_PATH

ENV = ENVIRONMENT

if __name__ == '__main__':
    print('CONFIG_PATH: {}'.format(CONFIG_PATH))
