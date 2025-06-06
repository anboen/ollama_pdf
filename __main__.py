import logging
import sys
from pathlib import Path
from datetime import datetime
from ollama_pdf.services import LLMServiceFactory
from ollama_pdf.utils import LLMConfigReader

format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
logging.basicConfig(
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    format=format,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # init
    config_path = Path("./config.yaml")
    folder_path = Path("./data")
    today = (
        Path("./results") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    logger.info("read config")
    config = LLMConfigReader(config_path)

    llm_service = LLMServiceFactory.createOpenAI(
        config.base_url, config.api_key, config.model, config.prompt
    )

    # get files
    with open(today, "a") as result_file:
        for file_path in folder_path.glob("*.pdf"):
            response = llm_service.serialize_pdf(file_path)
            logger.info(f"File {file_path} done")
            result_file.write(f"{file_path}:\n")
            result_file.write(response)
            result_file.write("\n\n")
