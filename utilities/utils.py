import os
from dotenv import load_dotenv
load_dotenv()

start_sequence = "\033[90m"
end_sequnece = "\033[0m"

def log(text):
    if os.getenv("DEBUG") == "True":
        print(f"{start_sequence}{text}{end_sequnece}")