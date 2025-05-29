import logging

logging.basicConfig(
    level=logging.INFO,
    format='\033[94m[%(levelname)s] %(asctime)s:\033[0m %(message)s',
    datefmt='%H:%M:%S'
)