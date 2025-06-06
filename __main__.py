import logging
import sys
from pathlib import Path
from datetime import datetime
from ollama_pdf.services import LLMServiceFactory
from ollama_pdf.utils import EnvConfigReader

format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
logging.basicConfig(
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    format=format,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # init paths
    input_path = Path("./data")
    output_path = Path("./results")
    today = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_file = output_path / today

    # create folders if necessary
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # read config
    logger.info("read config")
    config = EnvConfigReader()

    llm_service = LLMServiceFactory.createOpenAI(
        config.base_url, config.api_key, config.model, config.prompt
    )

    # get files
    with open(out_file, "a") as result_file:
        for file_path in input_path.glob("*.pdf"):
            response = llm_service.serialize_pdf(file_path)
            logger.info(f"File {file_path} done")
            result_file.write(f"{file_path}:\n")
            result_file.write(response)
            result_file.write("\n\n")
