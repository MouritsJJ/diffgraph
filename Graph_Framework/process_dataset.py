import importlib
from utils.util import parse_args

def process(config):
    # Preprocess dataset
    name = config["name"]
    dataset_class = getattr(importlib.import_module(f"datasets.{name.lower()}"), name)
    dataset_class(config)

if __name__ == '__main__':
    args = parse_args()
    process(args['dataset'])
