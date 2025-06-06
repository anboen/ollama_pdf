import yaml
from pathlib import Path


class LLMConfigReader:
    """Object to read an store configs"""

    def __init__(self, config_path: Path):
        """Reads config into memory

        Args:
            config_path (Path): _description_
        """
        with open(config_path, "r") as config_file:
            self._config = yaml.safe_load(config_file)
        prompt_path = Path(self._config["prompt_file"])
        with open(prompt_path, "r") as prompt_file:
            self._prompt = prompt_file.read()

    @property
    def base_url(self):
        return self._config["openAI"]["base_url"]

    @property
    def model(self):
        return self._config["openAI"]["model"]

    @property
    def api_key(self):
        return self._config["openAI"]["api_key"]

    @property
    def prompt(self):
        return self._prompt
