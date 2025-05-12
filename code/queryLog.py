# from datetime import datetime
# import json
# import os
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
# import io

# SCOPES = ['https://www.googleapis.com/auth/drive.file']  # Limited to file access
# CREDENTIALS_FILE = 'credentials.json'
# DRIVE_FOLDER_ID = '1YyDtIdHV0LjuljMZ4ygYgyfIsWnozYN6'  # Replace with your folder ID

# def get_drive_service():
#     creds = None
#     if os.path.exists('token.json'):
#         creds = Credentials.from_authorized_user_file('token.json', SCOPES)
#     if not creds or not creds.valid:
#         flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
#         creds = flow.run_local_server(port=0)
#         with open('token.json', 'w') as token:
#             token.write(creds.to_json())
#     return build('drive', 'v3', credentials=creds)

# def log_entry(index, message, data=None, status="info"):
#     """Log to Google Drive instead of local storage."""
#     entry = {
#         "timestamp": datetime.now().isoformat(),
#         "index": index,
#         "status": status,
#         "message": message,
#         "data": data
#     }
    
#     try:
#         # Create a temporary local file
#         temp_file = "temp_log.jsonl"
#         with open(temp_file, "a") as f:
#             f.write(json.dumps(entry) + "\n")
        
#         # Upload to Google Drive
#         drive_service = get_drive_service()
#         file_metadata = {
#             'name': 'data_ingestion_log.jsonl',
#             'parents': [DRIVE_FOLDER_ID]
#         }
#         media = MediaFileUpload(temp_file, mimetype='application/jsonl')
#         file = drive_service.files().create(
#             body=file_metadata,
#             media_body=media,
#             fields='id'
#         ).execute()
        
#         # Cleanup temp file
#         os.remove(temp_file)
        
#     except Exception as e:
#         print(f"Google Drive upload failed: {str(e)}")
#         # Fallback to local save
#         with open("data_ingestion_log.jsonl", "a") as log_file:
#             log_file.write(json.dumps(entry) + "\n")

# if __name__ == "__main__":
#     log_entry(1, "test")

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