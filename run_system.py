#!/usr/bin/env python3
"""
🚀 SYSTEM RUNNER - Reddit Mention Tracker
==========================================

This script starts the complete Reddit Mention Tracker system and opens the web interface.
Use this to run and interact with the system.

Usage:
    python run_system.py                    # Start full system with web UI
    python run_system.py --api-only        # Start API server only
    python run_system.py --ui-only         # Start UI only (no API)
    python run_system.py --port 7860       # Custom port
    python run_system.py --debug           # Debug mode
    python run_system.py --no-browser      # Don't auto-open browser
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal
import threading
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import psutil


class SystemRunner:
    """Manages the complete Reddit Mention Tracker system."""
    
    def __init__(self, project_root: str = None):
        """Initialize the system runner."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.processes = []
        self.running = False
        
        # Ensure we're in the right directory
        os.chdir(self.project_root)
        
        print("[START] REDDIT MENTION TRACKER - SYSTEM RUNNER")
        print("=" * 50)
        print(f"[ROOT] Project Root: {self.project_root}")
        print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
    
    def check_system_requirements(self) -> bool:
        """Check if system requirements are met."""
        print("\n[CHECK] CHECKING SYSTEM REQUIREMENTS...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            print("[ERROR] Python 3.11+ required")
            return False
        print(f"[OK] Python {sys.version.split()[0]}")
        
        # Check required files
        required_files = [
            'app.py',
            'requirements.txt',
            'database/models.py',
            'scraper/reddit_scraper.py',
            'analytics/metrics_analyzer.py',
            'ui/visualization.py'
        ]
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                print(f"[ERROR] Missing required file: {file_path}")
                return False
        print("[OK] All required files present")
        
        # Check if main modules are importable
        try:
            sys.path.insert(0, str(self.project_root))
            
            # Test imports
            import database.models
            import scraper.reddit_scraper
            import analytics.metrics_analyzer
            import ui.visualization
            
            print("[OK] All core modules importable")
            
        except ImportError as e:
            print(f"[ERROR] Import error: {e}")
            print("   Try: pip install -r requirements.txt")
            return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        print("\n[DEPS] CHECKING DEPENDENCIES...")
        
        # Check if requirements.txt exists
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            print("[ERROR] requirements.txt not found")
            return False
        
        try:
            # Try to install requirements
            print("Installing dependencies from requirements.txt...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("[OK] Dependencies installed successfully")
                return True
            else:
                print("[WARN] Some dependencies failed to install, trying essential ones...")
                
                # Try installing essential dependencies one by one
                essential_deps = [
                    'gradio>=4.0.0',
                    'fastapi>=0.100.0', 
                    'uvicorn>=0.20.0',
                    'pandas>=2.0.0',
                    'numpy>=1.24.0',
                    'plotly>=5.15.0',
                    'sqlalchemy>=2.0.0',
                    'textblob>=0.17.0',
                    'requests>=2.31.0',
                    'python-dotenv>=1.0.0',
                    'psutil>=5.9.0',
                    'pydantic>=2.5.0'
                ]
                
                failed_deps = []
                for dep in essential_deps:
                    try:
                        subprocess.run([
                            sys.executable, '-m', 'pip', 'install', dep
                        ], check=True, capture_output=True, timeout=60)
                        print(f"[OK] Installed: {dep}")
                    except subprocess.CalledProcessError:
                        failed_deps.append(dep)
                        print(f"[WARN] Failed: {dep}")
                
                if len(failed_deps) < len(essential_deps) / 2:  # If more than half succeeded
                    print("[OK] Essential dependencies installed (some optional failed)")
                    return True
                else:
                    print(f"[ERROR] Too many essential dependencies failed: {failed_deps}")
                    return False
            
        except subprocess.TimeoutExpired:
            print("[ERROR] Dependency installation timed out")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to install dependencies: {e}")
            print("   Try manually: pip install -r requirements.txt")
            return False
    
    def setup_database(self) -> bool:
        """Setup and initialize the database."""
        print("\n[DB] SETTING UP DATABASE...")
        
        try:
            from database.models import DatabaseManager
            
            # Create data directory if it doesn't exist
            data_dir = self.project_root / 'data'
            data_dir.mkdir(exist_ok=True)
            
            # Initialize database
            db_path = data_dir / 'reddit_mentions.db'
            db_manager = DatabaseManager(f"sqlite:///{db_path}")
            db_manager.create_tables()
            db_manager.close()
            
            print(f"[OK] Database initialized: {db_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Database setup failed: {e}")
            return False
    
    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def find_available_port(self, start_port: int = 7860) -> int:
        """Find an available port starting from start_port."""
        port = start_port
        while port < start_port + 100:  # Try 100 ports
            if self.check_port_availability(port):
                return port
            port += 1
        
        raise RuntimeError(f"No available ports found starting from {start_port}")
    
    def start_api_server(self, port: int = 8000) -> subprocess.Popen:
        """Start the API server."""
        print(f"\n🌐 STARTING API SERVER on port {port}...")
        
        try:
            # Check if API endpoints exist
            api_file = self.project_root / 'api' / 'endpoints.py'
            if api_file.exists():
                cmd = [
                    sys.executable, '-m', 'uvicorn', 
                    'api.endpoints:app',
                    '--host', '0.0.0.0',
                    '--port', str(port),
                    '--reload'
                ]
                
                process = subprocess.Popen(
                    cmd,
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait a moment to see if it starts successfully
                time.sleep(2)
                if process.poll() is None:
                    print(f"✅ API Server started on http://localhost:{port}")
                    return process
                else:
                    print("❌ API Server failed to start")
                    return None
            else:
                print("⚠️  API endpoints not found, skipping API server")
                return None
                
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return None
    
    def start_gradio_interface(self, port: int = 7860, debug: bool = False, 
                              share: bool = False) -> subprocess.Popen:
        """Start the Gradio web interface."""
        print(f"\n🎨 STARTING WEB INTERFACE on port {port}...")
        
        try:
            # Create comprehensive startup script with detailed logging
            startup_script = f"""
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
logger.info(f"Python version: {{sys.version}}")
logger.info(f"Working directory: {{os.getcwd()}}")
logger.info(f"Python path: {{sys.path[:3]}}")

# Set up path and directory
sys.path.insert(0, r'{self.project_root}')
os.chdir(r'{self.project_root}')
logger.info(f"Changed to directory: {{os.getcwd()}}")

try:
    logger.info("[IMPORT] Importing required modules...")
    
    # Test critical imports first
    import gradio as gr
    logger.info(f"[OK] Gradio imported successfully: {{gr.__version__}}")
    
    import pandas as pd
    logger.info(f"[OK] Pandas imported: {{pd.__version__}}")
    
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
    logger.info(f"[OK] Tracker created successfully in {{init_time:.2f}}s")
    
    logger.info("[UI] Creating Gradio interface...")
    start_time = time.time()
    interface = tracker.create_gradio_interface()
    interface_time = time.time() - start_time
    logger.info(f"[OK] Interface created successfully in {{interface_time:.2f}}s")
    
    logger.info("[LAUNCH] Launching Gradio interface...")
    logger.info(f"   Server port: {port}")
    logger.info(f"   Server name: 0.0.0.0")
    logger.info(f"   Debug mode: {debug}")
    logger.info(f"   Share: {share}")
    
    # Test port availability before launch
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('localhost', {port}))
        if result == 0:
            logger.error(f"[ERROR] Port {port} is already in use!")
            raise Exception(f"Port {port} is already in use")
        else:
            logger.info(f"[OK] Port {port} is available")
    
    # Launch with detailed monitoring
    launch_start = time.time()
    logger.info("[START] Starting Gradio launch...")
    
    interface.launch(
        server_port={port},
        server_name="0.0.0.0",
        debug={debug},
        share={share},
        show_error=True,
        quiet=False,
        prevent_thread_lock=False,
        show_tips=True,
        enable_queue=True
    )
    
    launch_time = time.time() - launch_start
    logger.info(f"[OK] Gradio launch completed in {{launch_time:.2f}}s")
    logger.info("[SUCCESS] GRADIO INTERFACE STARTED SUCCESSFULLY!")
    
except ImportError as e:
    logger.error(f"[ERROR] Import error: {{e}}")
    logger.error("[INFO] Available modules in current directory:")
    try:
        for item in os.listdir('.'):
            if os.path.isdir(item) and not item.startswith('.'):
                logger.error(f"   [DIR] {{item}}")
    except:
        pass
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    logger.error(f"[ERROR] Failed to start interface: {{e}}")
    logger.error("[INFO] Error details:")
    traceback.print_exc()
    
    # Additional debugging info
    logger.error("[DEBUG] System state debugging:")
    logger.error(f"   Current working directory: {{os.getcwd()}}")
    logger.error(f"   Python executable: {{sys.executable}}")
    logger.error(f"   Environment variables:")
    for key in ['PYTHONPATH', 'PATH', 'GRADIO_SERVER_PORT']:
        value = os.environ.get(key, 'Not set')
        logger.error(f"     {{key}}: {{value}}")
    
    sys.exit(1)
"""
            
            # Write startup script with error handling
            temp_script = self.project_root / 'gradio_startup_debug.py'
            try:
                with open(temp_script, 'w', encoding='utf-8') as f:
                    f.write(startup_script)
                print(f"✅ Startup script written to: {temp_script}")
            except Exception as e:
                print(f"❌ Failed to write startup script: {e}")
                return None
            
            # Start the process with enhanced monitoring
            print("🚀 Starting Gradio process...")
            try:
                process = subprocess.Popen(
                    [sys.executable, str(temp_script)],
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
                )
                print(f"✅ Process started with PID: {process.pid}")
                
            except Exception as e:
                print(f"❌ Failed to start process: {e}")
                return None
            
            # Enhanced startup monitoring
            print("⏳ Monitoring startup process...")
            startup_timeout = 120  # 2 minutes for complex initialization
            output_lines = []
            last_output_time = time.time()
            port_check_interval = 5
            last_port_check = 0
            
            for i in range(startup_timeout):
                time.sleep(1)
                
                # Check if process died
                if process.poll() is not None:
                    print("❌ Process terminated unexpectedly!")
                    
                    # Get remaining output
                    try:
                        remaining_output, _ = process.communicate(timeout=3)
                        if remaining_output:
                            output_lines.extend(remaining_output.strip().split('\n'))
                    except:
                        pass
                    
                    # Show process output
                    print("📋 Process output (last 30 lines):")
                    for line in output_lines[-30:]:
                        if line.strip():
                            print(f"   {line}")
                    
                    # Check log file
                    log_file = self.project_root / 'gradio_startup.log'
                    if log_file.exists():
                        print("📋 Startup log file content:")
                        try:
                            with open(log_file, 'r') as f:
                                log_content = f.read()
                                print(log_content[-2000:])  # Last 2000 chars
                        except Exception as e:
                            print(f"   Could not read log file: {e}")
                    
                    return None
                
                # Try to read output in real-time
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        line = line.strip()
                        if line:
                            output_lines.append(line)
                            print(f"   📝 {line}")
                            last_output_time = time.time()
                            
                            # Check for success/error indicators
                            if "Running on local URL" in line or "GRADIO INTERFACE STARTED SUCCESSFULLY" in line:
                                print("   🎯 Success indicator detected!")
                            elif "Error:" in line or "Failed:" in line or "Exception:" in line:
                                print(f"   ❌ Error detected: {line}")
                            elif "✅" in line:
                                print(f"   ✅ Progress: {line}")
                                
                except Exception as e:
                    # No more output available right now
                    pass
                
                # Periodic port checking
                current_time = time.time()
                if current_time - last_port_check >= port_check_interval:
                    last_port_check = current_time
                    
                    if self.check_port_in_use(port):
                        print(f" Port {port} is now active!")
                        
                        # Double-check with HTTP request
                        if self.test_http_connection(port):
                            print(f" Web Interface started successfully on http://localhost:{port}")
                            
                            # Clean up temp files
                            try:
                                if temp_script.exists():
                                    temp_script.unlink()
                            except:
                                pass
                            
                            return process
                        else:
                            print("     Port is active but HTTP connection failed")
                
                # Progress reporting
                if i % 15 == 0 and i > 0:
                    print(f"   ⏱  Still starting... ({i}/{startup_timeout}s)")
                    if output_lines:
                        recent_lines = [line for line in output_lines[-5:] if line.strip()]
                        if recent_lines:
                            print(f"    Recent: {recent_lines[-1][:80]}...")
                
                # Check for output stall
                if current_time - last_output_time > 45 and i > 30:
                    print(f"     No output for {int(current_time - last_output_time)}s")
                    
                    # Try to get process status
                    try:
                        if process.poll() is None:
                            print("    Process is still running but not producing output")
                            # Check if gradio might be waiting for user input
                            if self.check_port_in_use(port):
                                print("    Port is active despite no recent output!")
                                if self.test_http_connection(port):
                                    print(f" Web Interface is actually working on http://localhost:{port}")
                                    return process
                    except Exception as e:
                        print(f"    Error checking process status: {e}")
            
            # Timeout reached
            print(" Startup timeout reached!")
            
            # Final diagnostics
            print(" Final diagnostics:")
            print(f"   Process status: {'Running' if process.poll() is None else 'Terminated'}")
            print(f"   Port {port} status: {'Active' if self.check_port_in_use(port) else 'Inactive'}")
            
            # Check log file one more time
            log_file = self.project_root / 'gradio_startup.log'
            if log_file.exists():
                print(" Final log file content:")
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        print(log_content[-1500:])  # Last 1500 chars
                except Exception as e:
                    print(f"   Could not read log file: {e}")
            
            # Terminate if still running
            if process.poll() is None:
                print(" Terminating hanging process...")
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except:
                    try:
                        process.kill()
                    except:
                        pass
            
            return None
            
        except Exception as e:
            print(f" Failed to start web interface: {e}")
            traceback.print_exc()
            return None
    
    def test_http_connection(self, port: int) -> bool:
        """Test HTTP connection to the Gradio interface."""
        try:
            import urllib.request
            import urllib.error
            
            url = f"http://localhost:{port}"
            request = urllib.request.Request(url)
            
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.getcode() == 200:
                    return True
            return False
            
        except Exception as e:
            print(f"    HTTP test failed: {e}")
            return False
    
    def check_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use."""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                return result == 0  # 0 means connection successful (port in use)
        except:
            return False
    
    def start_monitoring(self) -> None:
        """Start system monitoring in background."""
        def monitor():
            while self.running:
                try:
                    # Check system resources
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    if cpu_percent > 90:
                        print(f"  High CPU usage: {cpu_percent:.1f}%")
                    
                    if memory.percent > 90:
                        print(f"  High memory usage: {memory.percent:.1f}%")
                    
                    # Check if processes are still running
                    for i, process in enumerate(self.processes):
                        if process and process.poll() is not None:
                            print(f"  Process {i} has stopped unexpectedly")
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception:
                    pass  # Ignore monitoring errors
        
        if psutil:
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
    
    def open_browser(self, url: str, delay: int = 3) -> None:
        """Open browser to the application URL."""
        def open_delayed():
            time.sleep(delay)
            try:
                webbrowser.open(url)
                print(f" Opened browser to {url}")
            except Exception as e:
                print(f"  Could not open browser: {e}")
                print(f"   Please manually open: {url}")
        
        browser_thread = threading.Thread(target=open_delayed, daemon=True)
        browser_thread.start()
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run_system(self, ui_port: int = 7860, api_port: int = 8000,
                   debug: bool = False, api_only: bool = False, 
                   ui_only: bool = False, no_browser: bool = False,
                   share: bool = False, use_subprocess: bool = False) -> None:
        """Run the complete system."""
        
        self.running = True
        self.setup_signal_handlers()
        
        try:
            # Find available ports
            if not ui_only:
                try:
                    api_port = self.find_available_port(api_port)
                except RuntimeError:
                    print(" No available ports for API server")
                    return
            
            if not api_only:
                try:
                    ui_port = self.find_available_port(ui_port)
                except RuntimeError:
                    print(" No available ports for web interface")
                    return
            
            # Start API server
            if not ui_only:
                api_process = self.start_api_server(api_port)
                if api_process:
                    self.processes.append(api_process)
            
            # Start web interface
            if not api_only:
                if not use_subprocess:  # Direct launch is default (unless --subprocess is used)
                    # Use direct launch (no subprocess) - DEFAULT BEHAVIOR
                    print("[INFO] Using direct launch method (recommended)")
                    success = self.start_direct_interface(ui_port, debug, share)
                    if not success:
                        print("[ERROR] Direct launch failed!")
                        return
                else:
                    # Use subprocess method (only when --subprocess flag is used)
                    print("[INFO] Using subprocess launch method")
                    ui_process = self.start_gradio_interface(ui_port, debug, share)
                    if ui_process:
                        self.processes.append(ui_process)
                        
                        # Open browser
                        if not no_browser:
                            self.open_browser(f"http://localhost:{ui_port}")
                    else:
                        print("[ERROR] Subprocess launch failed!")
                        return
            
            # Start monitoring
            self.start_monitoring()
            
            # Print status
            print("\n SYSTEM STARTED SUCCESSFULLY!")
            print("=" * 50)
            
            if not ui_only and any(p for p in self.processes if p):
                print(f" API Server: http://localhost:{api_port}")
                print(f"    API Docs: http://localhost:{api_port}/docs")
            
            if not api_only and any(p for p in self.processes if p):
                print(f" Web Interface: http://localhost:{ui_port}")
                print(f"    API Docs: http://localhost:{api_port}/docs")
            
            print("\n AVAILABLE FEATURES:")
            print("   • Search Reddit mentions")
            print("   • Real-time analytics dashboard")
            print("   • Data visualization")
            print("   • Export functionality")
            print("   • System monitoring")
            
            print("\n Press Ctrl+C to stop the system")
            print("=" * 50)
            
            # Keep running until interrupted
            try:
                while self.running and any(p and p.poll() is None for p in self.processes):
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            print(f"💥 System startup failed: {e}")
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Gracefully shutdown all processes."""
        print("\n SHUTTING DOWN SYSTEM...")
        
        self.running = False
        
        # Terminate all processes
        for i, process in enumerate(self.processes):
            if process and process.poll() is None:
                try:
                    print(f"   Stopping process {i}...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"   Force killing process {i}...")
                        process.kill()
                        
                except Exception as e:
                    print(f"   Error stopping process {i}: {e}")
        
        # Clean up temporary files
        temp_files = [
            'temp_gradio_launcher.py',
            'validation_test.db'
        ]
        
        for temp_file in temp_files:
            temp_path = self.project_root / temp_file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
        
        print(" System shutdown complete")
    
    def run_quick_test(self) -> bool:
        """Run a quick test of core components."""
        print("\n[TEST] RUNNING QUICK SYSTEM TEST...")
        
        try:
            # Test database
            from database.models import DatabaseManager
            db_manager = DatabaseManager()
            print("[OK] Database test passed")
            
            # Test scraper
            from scraper.reddit_scraper import RedditScraper
            scraper = RedditScraper(db_manager)
            print("[OK] Scraper test passed")
            
            # Test analytics
            from analytics.metrics_analyzer import MetricsAnalyzer
            analyzer = MetricsAnalyzer(db_manager)
            print("[OK] Analytics test passed")
            
            # Test UI
            from ui.visualization import MetricsVisualizer
            viz = MetricsVisualizer()
            print("[OK] UI test passed")
            
            # Clean up
            db_manager.close()
            
            # Test complete
            print("[SUCCESS] Quick test completed successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Quick test failed: {e}")
            return False
    
    def debug_direct_interface(self, port: int = 7860) -> bool:
        """Debug the interface by running it directly in this process (for testing only)."""
        print(f"\n🔬 DIRECT INTERFACE DEBUG TEST on port {port}...")
        
        try:
            # Import and test directly
            print(" Testing imports...")
            sys.path.insert(0, str(self.project_root))
            
            from app import EnhancedRedditMentionTracker
            print(" EnhancedRedditMentionTracker imported")
            
            print("  Creating tracker instance...")
            tracker = EnhancedRedditMentionTracker()
            print(" Tracker created successfully")
            
            print(" Creating Gradio interface...")
            interface = tracker.create_gradio_interface()
            print(" Interface created successfully")
        
            print(f" Testing interface launch on port {port}...")
            
            # Try launching with minimal settings for testing
            try:
                # This will block, so we're just testing if it would work
                print("   Testing launch parameters...")
                
                # Check if port is available
                if not self.check_port_availability(port):
                    print(f"    Port {port} is not available")
                    return False
            
                print("    Port is available")
                print("    Attempting launch (will terminate quickly for testing)...")
                
                # Launch with a thread to avoid blocking
                import threading
                import time
                
                launch_result = {"success": False, "error": None}
                
                def launch_test():
                    try:
                        interface.launch(
                            server_port=port,
                            server_name="127.0.0.1",  # localhost only for testing
                            debug=True,
                            share=False,
                            show_error=True,
                            quiet=False,
                            prevent_thread_lock=True  # Don't block
                        )
                        launch_result["success"] = True
                    except Exception as e:
                        launch_result["error"] = str(e)
                
                # Start launch in thread
                launch_thread = threading.Thread(target=launch_test, daemon=True)
                launch_thread.start()
                
                # Wait and check
                for i in range(10):
                    time.sleep(1)
                    if self.check_port_in_use(port):
                        print(f"    Interface is responding on port {port}!")
                        if self.test_http_connection(port):
                            print("    HTTP connection successful!")
                            print("   Direct interface test PASSED")
                            
                            # Try to gracefully stop
                            try:
                                interface.close()
                            except:
                                pass
                            
                            return True
                        else:
                            print("     Port active but HTTP failed")
                    
                    if launch_result["error"]:
                        print(f"    Launch error: {launch_result['error']}")
                        return False
                
                print("   Interface didn't become responsive in 10 seconds")
                return False
                
            except Exception as e:
                print(f" Launch test failed: {e}")
                return False
                
        except Exception as e:
            print(f" Direct interface test failed: {e}")
            traceback.print_exc()
            return False
    
    def start_direct_interface(self, port: int = 7860, debug: bool = False, share: bool = False) -> bool:
        """Start Gradio interface directly in this process (no subprocess)."""
        
        # Find an available port if the default is taken
        try:
            available_port = self.find_available_port(port)
            if available_port != port:
                print(f"[INFO] Port {port} in use, using port {available_port} instead")
                port = available_port
        except RuntimeError as e:
            print(f"[ERROR] Could not find available port: {e}")
            return False
        
        print(f"\n[DIRECT] STARTING WEB INTERFACE DIRECTLY on port {port}...")
        
        try:
            # Set up environment
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.chdir(self.project_root)
            sys.path.insert(0, str(self.project_root))
            
            print("[IMPORT] Importing application...")
            from app import EnhancedRedditMentionTracker
            
            print("[CREATE] Creating tracker...")
            tracker = EnhancedRedditMentionTracker()
            
            print("[UI] Creating interface...")
            interface = tracker.create_gradio_interface()
            
            print(f"[LAUNCH] Launching on port {port}...")
            
            # Launch interface
            interface.launch(
                server_port=port,
                server_name="0.0.0.0",
                debug=debug,
                share=share,
                show_error=True,
                quiet=False,
                prevent_thread_lock=False,
                inbrowser=False  # Don't auto-open browser
            )
            
            print(f"[SUCCESS] Interface launched successfully on http://localhost:{port}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Direct launch failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="System Runner for Reddit Mention Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_system.py                    # Start full system
  python run_system.py --port 8080       # Custom UI port
  python run_system.py --api-only        # API server only
  python run_system.py --ui-only         # Web interface only
  python run_system.py --debug           # Debug mode
  python run_system.py --no-browser      # Don't open browser
  python run_system.py --share           # Create public Gradio link
  python run_system.py --debug-direct    # Direct interface debug test
        """
    )
    
    parser.add_argument('--port', type=int, default=7860,
                       help='Port for web interface (default: 7860)')
    parser.add_argument('--api-port', type=int, default=8000,
                       help='Port for API server (default: 8000)')
    parser.add_argument('--api-only', action='store_true',
                       help='Start API server only')
    parser.add_argument('--ui-only', action='store_true',
                       help='Start web interface only')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not automatically open browser')
    parser.add_argument('--share', action='store_true',
                       help='Create public Gradio share link')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick system test only')
    parser.add_argument('--debug-direct', action='store_true',
                       help='Run direct interface debug test')
    parser.add_argument('--subprocess', action='store_true',
                       help='Use subprocess launch instead of direct (not recommended)')
    parser.add_argument('--project-root',
                       help='Project root directory (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        # Initialize system runner
        runner = SystemRunner(args.project_root)
        
        # Quick test mode
        if args.quick_test:
            success = runner.run_quick_test()
            sys.exit(0 if success else 1)
        
        # Direct debug mode
        if args.debug_direct:
            print("\n🔬 RUNNING DIRECT INTERFACE DEBUG...")
            
            # Basic checks first
            if not runner.check_system_requirements():
                print(" System requirements not met")
                sys.exit(1)
            
            # Try direct interface test
            success = runner.debug_direct_interface(args.port)
            if success:
                print("\n Direct interface test PASSED - the issue is likely in subprocess handling")
                print("   Recommendation: Check subprocess creation, logging, or process communication")
            else:
                print("\n Direct interface test FAILED - the issue is in the core application")
                print("   Recommendation: Check application initialization or Gradio setup")
            
            sys.exit(0 if success else 1)
        
        # Check system requirements
        if not runner.check_system_requirements():
            print("\n System requirements check failed!")
            print("Please ensure all required files and dependencies are present.")
            sys.exit(1)
        
        # Install dependencies
        if not runner.install_dependencies():
            print("\n Dependency installation failed!")
            sys.exit(1)
        
        # Setup database
        if not runner.setup_database():
            print("\n Database setup failed!")
            sys.exit(1)
        
        # Run quick validation test
        print("\n[TEST] Running quick validation test...")
        if runner.run_quick_test():
            print("[OK] Quick test passed!")
        else:
            print("\n[WARN] Quick test failed, but continuing...")
        
        # Additional debug mode checks
        if args.debug:
            print("\n DEBUG MODE - Running additional diagnostics...")
            
            # Test direct interface first
            print("🔬 Testing direct interface capabilities...")
            direct_success = runner.debug_direct_interface(args.port)
            
            if direct_success:
                print(" Direct interface test passed - subprocess should work")
            else:
                print(" Direct interface test failed - will likely fail in subprocess too")
                print("   Continuing anyway for comprehensive debugging...")
            
            # Check port availability
            print(f"🔍 Checking port {args.port} availability...")
            if runner.check_port_availability(args.port):
                print(f" Port {args.port} is available")
            else:
                print(f" Port {args.port} is in use, finding alternative...")
                try:
                    alt_port = runner.find_available_port(args.port)
                    print(f" Alternative port found: {alt_port}")
                    args.port = alt_port
                except RuntimeError:
                    print(" No available ports found!")
                    sys.exit(1)
        
        # Run the system
        runner.run_system(
            ui_port=args.port,
            api_port=args.api_port,
            debug=args.debug,
            api_only=args.api_only,
            ui_only=args.ui_only,
            no_browser=args.no_browser,
            share=args.share,
            use_subprocess=args.subprocess
        )
        
    except KeyboardInterrupt:
        print("\n System startup interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n System startup failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 