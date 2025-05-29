#!/usr/bin/env python3
"""
Simplified Reddit Mention Tracker for testing.
"""
import sys
import os
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

class SimpleRedditMentionTracker:
    """Simplified Reddit Mention Tracker for testing."""
    
    def __init__(self):
        print("Initializing Simple Reddit Mention Tracker...")
        
        # Basic settings
        self.app_name = "Reddit Mention Tracker"
        self.app_version = "1.0.0"
        
        # Initialize basic components
        try:
            from database.models import DatabaseManager
            self.db_manager = DatabaseManager("sqlite:///data/reddit_mentions.db")
            self.db_manager.create_tables()
            print("✅ Database initialized")
        except Exception as e:
            print(f"⚠️  Database initialization failed: {e}")
            self.db_manager = None
        
        print("✅ Simple Reddit Mention Tracker initialized")
    
    def create_gradio_interface(self) -> gr.Blocks:
        """Create simplified Gradio interface."""
        with gr.Blocks(
            title=self.app_name,
            theme=gr.themes.Soft()
        ) as interface:
            
            # Header
            gr.Markdown(f"""
            # {self.app_name} v{self.app_version}
            
            Simplified Reddit mention tracking interface for testing.
            """)
            
            with gr.Tabs():
                # Main search tab
                with gr.Tab("Search & Analytics"):
                    with gr.Row():
                        with gr.Column():
                            search_term = gr.Textbox(
                                label="Search Term",
                                placeholder="Enter keywords to search for",
                                lines=1
                            )
                            
                            search_btn = gr.Button("Start Search", variant="primary")
                            
                            search_status = gr.Textbox(
                                label="Search Status",
                                value="Ready to search",
                                interactive=False
                            )
                        
                        with gr.Column():
                            search_history = gr.Dataframe(
                                headers=["Search Term", "Status"],
                                label="Search History",
                                interactive=False
                            )
                
                with gr.Tab("Analytics"):
                    analytics_plot = gr.Plot(label="Analytics")
                
                with gr.Tab("System Status"):
                    status_html = gr.HTML(
                        value="<p>System Status: Ready</p>",
                        label="Status"
                    )
            
            # Event handlers
            def handle_search(search_term):
                """Handle search request."""
                try:
                    if not search_term:
                        return "Please enter a search term", pd.DataFrame()
                    
                    # Simulate search
                    status = f"Search completed for: {search_term}"
                    
                    # Create sample history
                    history_data = [
                        [search_term, "Completed"]
                    ]
                    df = pd.DataFrame(history_data, columns=["Search Term", "Status"])
                    
                    return status, df
                    
                except Exception as e:
                    return f"Search failed: {str(e)}", pd.DataFrame()
            
            # Wire up events
            search_btn.click(
                handle_search,
                inputs=[search_term],
                outputs=[search_status, search_history]
            )
        
        return interface

def main():
    """Main entry point."""
    try:
        print("Starting Simple Reddit Mention Tracker...")
        
        # Create application
        app = SimpleRedditMentionTracker()
        
        # Create interface
        interface = app.create_gradio_interface()
        
        # Launch
        print("Launching interface on http://localhost:7860...")
        interface.launch(
            server_port=7860,
            server_name="0.0.0.0",
            debug=False,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"Failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 