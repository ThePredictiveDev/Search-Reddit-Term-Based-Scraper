#!/usr/bin/env python3
"""
Minimal interface test to isolate startup issues.
"""
import sys
import os
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('minimal_test.log', mode='w')
    ]
)
logger = logging.getLogger('minimal_test')

def test_minimal_interface():
    """Test minimal interface startup."""
    try:
        logger.info("=== MINIMAL INTERFACE TEST STARTING ===")
        
        # Set up environment
        logger.info("Setting up environment...")
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Test imports
        logger.info("Testing imports...")
        
        logger.info("Importing gradio...")
        import gradio as gr
        logger.info(f"Gradio version: {gr.__version__}")
        
        logger.info("Importing app...")
        from app import EnhancedRedditMentionTracker
        logger.info("App imported successfully")
        
        # Create tracker
        logger.info("Creating tracker...")
        tracker = EnhancedRedditMentionTracker()
        logger.info("Tracker created successfully")
        
        # Create interface
        logger.info("Creating interface...")
        interface = tracker.create_gradio_interface()
        logger.info("Interface created successfully")
        
        # Launch interface
        logger.info("Launching interface...")
        interface.launch(
            server_port=7860,
            server_name="0.0.0.0",
            debug=True,
            share=False,
            show_error=True,
            quiet=False,
            prevent_thread_lock=False,
            inbrowser=False
        )
        
        logger.info("Interface launched successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_minimal_interface()
    if success:
        print("[SUCCESS] Minimal interface test passed!")
    else:
        print("[ERROR] Minimal interface test failed!")
        sys.exit(1) 