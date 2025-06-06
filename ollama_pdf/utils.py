import os
import yaml
import logging
from abc import ABC
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr


logger = logging.getLogger(__name__)


class BaseConfigReader(ABC):
    """Base class for config readers"""

    def __init__(self):
        """Reads config from environment variables"""
        self._config = {
            "openAI": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "api_key": "your_api_key_here",
            },
            "prompt": "Please summarize the following PDF content.",
        }

    def _load_prompt_file(self, prompt_path: Path):
        """Loads the prompt from a file specified in the config.

        Args:
            prompt_path (Path): Path to the prompt file.
        """

        if prompt_path.exists():
            with open(prompt_path, "r") as prompt_file:
                self._config["prompt"] = prompt_file.read()

    @property
    def base_url(self) -> str:
        """Returns the base URL to use for the LLM Service"""
        return self._config["openAI"]["base_url"]

    @property
    def model(self) -> str:
        """Returns the model to use for the LLM Service"""
        return self._config["openAI"]["model"]

    @property
    def api_key(self) -> SecretStr:
        """Returns the API key to use for the LLM Service"""
        return SecretStr(self._config["openAI"]["api_key"])

    @property
    def prompt(self) -> str:
        """Returns the prompt to use for the LLM"""
        return self._config["prompt"]


class YAMLConfigReader(BaseConfigReader):
    """Object to read an store configs"""

    def __init__(self, config_path: Path):
        """Reads config into memory

        Args:
            config_path (Path): path to yaml config file.
            If the file does not exist, a default config is created.
        """
        # Initialize default config
        super().__init__()

        # Check if the config file exists, if not, create a default config
        if not config_path.exists():
            # Read config from yaml file
            self._load_from_yaml(config_path)
            if "prompt_file" in self._config:
                self._load_prompt_file(Path(self._config["prompt_file"]))
            else:
                logger.warning(
                    f"Prompt file {self._config['prompt_file']} not found."
                )

        else:
            logger.warning(
                f"Config file {config_path} not found. Using default config."
            )

    def _load_from_yaml(self, config_path: Path):
        """Loads config from a YAML file.

        Args:
            config_path (Path): Path to the YAML config file.
        """
        with open(config_path, "r") as config_file:
            self._config = yaml.safe_load(config_file)


class EnvConfigReader(BaseConfigReader):
    """Object to read an store configs from environment variables"""

    def __init__(self):
        """Reads config from environment variables"""
        super().__init__()
        load_dotenv()
        self._load_env_vars()

    def _load_env_vars(self):
        """Loads config from environment variables."""
        self._config["openAI"]["base_url"] = os.getenv(
            "OPENAI_BASE_URL", self._config["openAI"]["base_url"]
        )
        self._config["openAI"]["model"] = os.getenv(
            "OPENAI_MODEL", self._config["openAI"]["model"]
        )
        self._config["openAI"]["api_key"] = os.getenv(
            "OPENAI_API_KEY", self._config["openAI"]["api_key"]
        )
        prompt_path = Path(os.getenv("PROMPT_FILE", ""))
        self._load_prompt_file(prompt_path)
