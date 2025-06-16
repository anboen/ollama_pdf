import logging
import sys
from pathlib import Path
from datetime import datetime
from ollama_pdf.services import LLMServiceFactory
from ollama_pdf.utils import EnvConfigReader
import pandas as pd
import time
import os

format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
logging.basicConfig(
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
    format=format,
)
logger = logging.getLogger(__name__)

ground_truth = {
    "2025-03-24 Garage W. Brunmann Rechnung für Mitsubishi Space Star.pdf": {
        "date": "24.03.2025",
        "reference": "00 00000 00000 00000 00004 50436",
        "amounts": [
            {"total": 644.6, "currency": "CHF"},
            {"total": 685.8, "currency": "EUR"},
        ],
        "taxes": [{"amount": 48.3, "rate": 8.1}],
        "address": {
            "street": "Steinäckerweg 18",
            "city": "Unterlunkhofen",
            "zip_code": "8918",
        },
        "IBAN": "CH83 3076 1648 2287 4200 1",
        "Errors": False,
    },
    "dummy.pdf": {
        "date": "",
        "reference": "",
        "amounts": [],
        "taxes": [],
        "address": {
            "street": "",
            "city": "",
            "zip_code": "",
        },
        "IBAN": "",
    },
    "invoice_170525657.pdf": {
        "date": "30.04.2025",
        "reference": "00 00005 08084 87540 40837 17145",
        "amounts": [{"total": 22.95, "currency": "CHF"}],
        "taxes": [{"amount": 1.27, "rate": 8.1}],
        "address": {
            "street": "Alte Tiefenaustrasse 6",
            "city": "Bern",
            "zip_code": "3050",
        },
        "IBAN": "CH21 3000 0002 3108 0014 1",
    },
    "Pre-Print_ESMAC2023_IntellEvent_vs_AC.pdf": {
        "date": "",
        "reference": "",
        "amounts": [],
        "taxes": [],
        "address": {
            "street": "",
            "city": "",
            "zip_code": "",
        },
        "IBAN": "",
    },
}


def _compare_response(result, ground_truth):
    """Compare the response with the ground truth."""
    checked = 0
    wrong = 0
    for key, value in ground_truth.items():
        checked += 1
        if key == "errors":
            continue

        if key not in result:
            wrong += 1
        elif result[key] != value:
            wrong += 1
        else:
            if isinstance(value, str):
                result_value = result[key].replace(" ", "")
                value = value.replace(" ", "")
                if result_value != value:
                    wrong += 1
            elif isinstance(value, list):
                for i in range(len(value)):
                    wrong_sub, checked_sub = _compare_response(
                        value[i], result[key][i]
                    )
                    wrong += wrong_sub
                    checked += checked_sub
            elif isinstance(value, dict):
                wrong_sub, checked_sub = _compare_response(value, result[key])
                wrong += wrong_sub
                checked += checked_sub
            else:
                if result[key] != value:
                    wrong += 1

    return wrong, checked


if __name__ == "__main__":
    # init paths
    today = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    input_path = Path("./data")
    output_file = Path("./results") / f"results_mean_{today}.csv"

    models = [
        "phi4:14b",
        "Qwen3:4b",
        "Qwen3:14b",
    ]
    chunk_sizes = [(800, 80), (750, 75), (850, 85)]

    # create folders if necessary
    input_path.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # read config
    logger.info("read config")
    config = EnvConfigReader()
    result_dict = {
        "accuracy": [],
        "model": [],
        "elapsed_time": [],
        "chunk_size": [],
    }

    for model in models:
        llm_service = LLMServiceFactory.create_service(
            config.service,
            config.base_url,
            config.api_key,
            model,
            config.embedding_model,
            config.prompt,
        )
        llm_service.load_model()

        for chunk_size, overlay_size in chunk_sizes:
            file_name = model.replace(":", "_")
            logger.warning(
                (f"Processing {model} " f"and {chunk_size}, {overlay_size}")
            )

            # get files
            for file_path in input_path.glob("*.pdf"):
                start_time = time.time()
                response = llm_service.extract_structure(
                    file_path,
                    chunk_size=chunk_size,
                    overlay_size=overlay_size,
                )
                delta_time = time.time() - start_time
                logger.warning(f"File {file_path}")

                wrong, checked = _compare_response(
                    response["result"].model_dump(),
                    ground_truth[file_path.name],
                )
                result_dict["accuracy"].append((checked - wrong) / checked)
                result_dict["model"].append(model)
                result_dict["elapsed_time"].append(delta_time)
                result_dict["chunk_size"].append(
                    f"{chunk_size}_{overlay_size}"
                )
            # save results to CSV
            if output_file.exists():
                logger.warning(f"Removing old file {output_file}")
                os.remove(output_file)

            result_df = pd.DataFrame(result_dict)
            result_df.groupby(["model", "chunk_size"]).mean().to_csv(
                output_file,
                index=True,
                sep=";",
            )
            logger.warning(f"Results saved to {output_file}")
