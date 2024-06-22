import subprocess
import logging
import os

# Configure logging
logging.basicConfig(filename='uvicorn_service.log', level=logging.DEBUG)

def run_uvicorn(port):
    try:
        # Command to start Uvicorn server
        command = f"uvicorn main:app --host 127.0.0.1 --port {port}"
        logging.info(f"Running command: {command}")
        # Start the subprocess
        subprocess.Popen(command, shell=True, cwd="C:/Users/tomas/Desktop/NLP")  # Use shell=True for Windows compatibility
        logging.info(f"Started Uvicorn on port {port}")
    except Exception as e:
        logging.error(f"Failed to start Uvicorn on port {port}: {e}")

# Define ports
ports = [8001, 8002, 8003, 8004]

# Start workers on different ports
for port in ports:
    run_uvicorn(port)
