
import sys
import os
import time
import traceback
import logging

# Configure logging for startup script
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gradio_startup.log', mode='w')
    ]
)
logger = logging.getLogger('startup')

logger.info("[START] GRADIO STARTUP SCRIPT STARTING")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path[:3]}")

# Set up path and directory
sys.path.insert(0, r'C:\Users\Devansh\Downloads\intern_application_round1')
os.chdir(r'C:\Users\Devansh\Downloads\intern_application_round1')
logger.info(f"Changed to directory: {os.getcwd()}")

try:
    logger.info("[IMPORT] Importing required modules...")
    
    # Test critical imports first
    import gradio as gr
    logger.info(f"[OK] Gradio imported successfully: {gr.__version__}")
    
    import pandas as pd
    logger.info(f"[OK] Pandas imported: {pd.__version__}")
    
    import plotly.graph_objects as go
    logger.info("[OK] Plotly imported successfully")
    
    # Import and create the application
    logger.info("[APP] Importing EnhancedRedditMentionTracker...")
    from app import EnhancedRedditMentionTracker
    logger.info("[OK] EnhancedRedditMentionTracker imported successfully")
    
    logger.info("[CREATE] Creating Reddit Mention Tracker instance...")
    start_time = time.time()
    tracker = EnhancedRedditMentionTracker()
    init_time = time.time() - start_time
    logger.info(f"[OK] Tracker created successfully in {init_time:.2f}s")
    
    logger.info("[UI] Creating Gradio interface...")
    start_time = time.time()
    interface = tracker.create_gradio_interface()
    interface_time = time.time() - start_time
    logger.info(f"[OK] Interface created successfully in {interface_time:.2f}s")
    
    logger.info("[LAUNCH] Launching Gradio interface...")
    logger.info(f"   Server port: 7860")
    logger.info(f"   Server name: 0.0.0.0")
    logger.info(f"   Debug mode: False")
    logger.info(f"   Share: False")
    
    # Test port availability before launch
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('localhost', 7860))
        if result == 0:
            logger.error(f"[ERROR] Port 7860 is already in use!")
            raise Exception(f"Port 7860 is already in use")
        else:
            logger.info(f"[OK] Port 7860 is available")
    
    # Launch with detailed monitoring
    launch_start = time.time()
    logger.info("[START] Starting Gradio launch...")
    
    interface.launch(
        server_port=7860,
        server_name="0.0.0.0",
        debug=False,
        share=False,
        show_error=True,
        quiet=False,
        prevent_thread_lock=False,
        show_tips=True,
        enable_queue=True
    )
    
    launch_time = time.time() - launch_start
    logger.info(f"[OK] Gradio launch completed in {launch_time:.2f}s")
    logger.info("[SUCCESS] GRADIO INTERFACE STARTED SUCCESSFULLY!")
    
except ImportError as e:
    logger.error(f"[ERROR] Import error: {e}")
    logger.error("[INFO] Available modules in current directory:")
    try:
        for item in os.listdir('.'):
            if os.path.isdir(item) and not item.startswith('.'):
                logger.error(f"   [DIR] {item}")
    except:
        pass
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    logger.error(f"[ERROR] Failed to start interface: {e}")
    logger.error("[INFO] Error details:")
    traceback.print_exc()
    
    # Additional debugging info
    logger.error("[DEBUG] System state debugging:")
    logger.error(f"   Current working directory: {os.getcwd()}")
    logger.error(f"   Python executable: {sys.executable}")
    logger.error(f"   Environment variables:")
    for key in ['PYTHONPATH', 'PATH', 'GRADIO_SERVER_PORT']:
        value = os.environ.get(key, 'Not set')
        logger.error(f"     {key}: {value}")
    
    sys.exit(1)
