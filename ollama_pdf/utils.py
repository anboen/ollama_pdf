import os
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
            "service": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-3.5-turbo",
            "embbedding_model": "text-embedding-ada-002",
            "api_key": "your_api_key_here",
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
    def service(self) -> str:
        """Returns the LLM Service to use"""
        return self._config["service"]

    @property
    def base_url(self) -> str:
        """Returns the base URL to use for the LLM Service"""
        return self._config["base_url"]

    @property
    def model(self) -> str:
        """Returns the model to use for the LLM Service"""
        return self._config["model"]

    @property
    def embedding_model(self) -> str:
        """Returns the embedding model to use for the LLM Service"""
        return self._config["embbedding_model"]

    @property
    def api_key(self) -> SecretStr:
        """Returns the API key to use for the LLM Service"""
        return SecretStr(self._config["api_key"])

    @property
    def prompt(self) -> str:
        """Returns the prompt to use for the LLM"""
        return self._config["prompt"]


class EnvConfigReader(BaseConfigReader):
    """Object to read an store configs from environment variables"""

    def __init__(self):
        """Reads config from environment variables"""
        super().__init__()
        load_dotenv()
        self._load_env_vars()

    def _load_env_vars(self):
        """Loads config from environment variables."""
        self._config["service"] = os.getenv("SERVICE", self._config["service"])
        self._config["base_url"] = os.getenv(
            "BASE_URL", self._config["base_url"]
        )
        self._config["model"] = os.getenv("MODEL", self._config["model"])
        self._config["embbedding_model"] = os.getenv(
            "EMBBEDDING_MODEL", self._config["embbedding_model"]
        )
        self._config["api_key"] = os.getenv("API_KEY", self._config["api_key"])
        prompt_path = Path(os.getenv("PROMPT_FILE", ""))
        self._load_prompt_file(prompt_path)
