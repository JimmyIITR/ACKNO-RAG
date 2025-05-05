from datetime import datetime
import json

def log_entry(index, message, data=None, status="info"):
    """Helper function to log messages and data to a JSONL file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "index": index,
        "status": status,
        "message": message,
        "data": data
    }
    try:
        with open("data_ingestion_log.jsonl", "a") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Failed to write log entry: {str(e)}")

if __name__ == "__main__":
    log_entry(1, "test")