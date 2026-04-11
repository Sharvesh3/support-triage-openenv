# main.py
import uvicorn
import os
from server.app import app

def main():
   
    port = int(os.environ.get("PORT", 8004))
    print(f"🚀 Starting Support Triage Pro Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()