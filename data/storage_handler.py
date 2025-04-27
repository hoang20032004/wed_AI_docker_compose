import json
import os
from datetime import datetime

class StorageHandler:
    def __init__(self):
        self.storage_path = os.path.join(os.path.dirname(__file__), 'storage.json')

    def save_data(self, data):
        try:
            storage = self.read_storage()
            storage['entries'].append({
                **data,
                'timestamp': datetime.now().isoformat()
            })
            storage['lastUpdated'] = datetime.now().isoformat()
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(storage, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

    def read_storage(self):
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {
                "entries": [],
                "lastUpdated": None,
                "metadata": {
                    "version": "1.0",
                    "description": "Data storage for AI web application"
                }
            }
