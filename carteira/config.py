from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

INPUT_FILE = DATA_DIR / "input_carteira.xlsx"
OUTPUT_FILE = DATA_DIR / "output_carteira.xlsx"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
