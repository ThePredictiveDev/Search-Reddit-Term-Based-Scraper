#!/usr/bin/env python3
"""
Basic Gradio test - minimal interface to test if Gradio works at all.
"""
import gradio as gr

def create_simple_interface():
    """Create a very simple Gradio interface."""
    
    def greet(name):
        return f"Hello {name}!"
    
    interface = gr.Interface(
        fn=greet,
        inputs=gr.Textbox(label="Your Name"),
        outputs=gr.Textbox(label="Greeting"),
        title="Simple Test Interface"
    )
    
    return interface

if __name__ == "__main__":
    print("Creating simple Gradio interface...")
    interface = create_simple_interface()
    
    print("Launching on http://localhost:7860...")
    interface.launch(
        server_port=7860,
        server_name="0.0.0.0",
        show_error=True,
        quiet=False
    ) 