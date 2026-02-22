#!/usr/bin/env python3
"""
Run script for Ultra Doc-Intelligence.
Starts both the API server and Streamlit UI.
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import streamlit
        import openai
        import chromadb
        print("[OK] All dependencies installed")
        return True
    except ImportError as e:
        print(f"[X] Missing dependency: {e.name}")
        print("  Run: pip install -r requirements.txt")
        return False


def check_env():
    """Check if environment is configured."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("XAI_API_KEY")
    if not api_key or api_key == "your_xai_api_key_here":
        print("[X] XAI_API_KEY not configured")
        print("  1. Copy .env.example to .env")
        print("  2. Add your xAI API key")
        return False
    
    print("[OK] Environment configured")
    return True


def run_api():
    """Start the FastAPI server."""
    print("\n[*] Starting API server on http://localhost:8000")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", 
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=Path(__file__).parent
    )


def run_ui():
    """Start the Streamlit UI."""
    print("[*] Starting UI on http://localhost:8501")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
         "--server.port", "8501", "--server.address", "0.0.0.0"],
        cwd=Path(__file__).parent
    )


def main():
    print("=" * 60)
    print("   Ultra Doc-Intelligence - Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_env():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    
    # Start services
    api_process = run_api()
    time.sleep(3)  # Wait for API to start
    ui_process = run_ui()
    
    print("\n" + "=" * 60)
    print("   Services Started!")
    print("=" * 60)
    print("\n  API:  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  UI:   http://localhost:8501")
    print("\n  Press Ctrl+C to stop all services")
    print("=" * 60 + "\n")
    
    # Open browser after a short delay
    time.sleep(2)
    webbrowser.open("http://localhost:8501")
    
    try:
        # Wait for processes
        api_process.wait()
    except KeyboardInterrupt:
        print("\n\n[*] Shutting down services...")
        api_process.terminate()
        ui_process.terminate()
        print("[OK] Services stopped")


if __name__ == "__main__":
    main()
