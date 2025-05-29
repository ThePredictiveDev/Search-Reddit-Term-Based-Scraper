#!/usr/bin/env python3
"""
Simple test script to verify Gradio interface works.
"""
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.getcwd())

def test_gradio_interface():
    """Test basic Gradio interface creation and startup."""
    try:
        print("Testing Gradio interface creation...")
        
        # Import the main app
        from app import EnhancedRedditMentionTracker
        
        print("✅ Successfully imported EnhancedRedditMentionTracker")
        
        # Create application instance
        print("Creating application instance...")
        tracker = EnhancedRedditMentionTracker()
        print("✅ Successfully created tracker instance")
        
        # Create Gradio interface
        print("Creating Gradio interface...")
        interface = tracker.create_gradio_interface()
        print("✅ Successfully created Gradio interface")
        
        # Launch interface
        print("Launching interface on http://localhost:7860...")
        interface.launch(
            server_port=7860,
            server_name="0.0.0.0",
            debug=False,
            share=False,
            show_error=True,
            quiet=False,
            prevent_thread_lock=False  # Allow it to block
        )
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_gradio_interface()
    sys.exit(0 if success else 1) 