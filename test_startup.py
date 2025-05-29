#!/usr/bin/env python3
"""
Comprehensive startup debug script to identify blocking issues.
"""
import sys
import os
import time
import traceback

# Add project root to path
sys.path.insert(0, os.getcwd())

def test_component(name, test_func):
    """Test a component and report results."""
    print(f"\n🧪 Testing {name}...")
    start_time = time.time()
    
    try:
        result = test_func()
        elapsed = time.time() - start_time
        print(f"✅ {name} - OK ({elapsed:.2f}s)")
        return True, result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {name} - FAILED ({elapsed:.2f}s): {e}")
        traceback.print_exc()
        return False, None

def test_basic_imports():
    """Test basic imports."""
    import gradio as gr
    import pandas as pd
    import plotly.graph_objects as go
    return "Basic imports successful"

def test_database_manager():
    """Test database manager."""
    from database.models import DatabaseManager
    db = DatabaseManager("sqlite:///test_debug.db")
    db.create_tables()
    session_id = db.create_search_session("test")
    db.close()
    # Clean up
    import os
    if os.path.exists("test_debug.db"):
        os.remove("test_debug.db")
    return f"Database manager - session {session_id} created"

def test_scraper():
    """Test scraper initialization."""
    from database.models import DatabaseManager
    from scraper.reddit_scraper import RedditScraper
    db = DatabaseManager("sqlite:///test_debug2.db")
    db.create_tables()
    scraper = RedditScraper(db)
    db.close()
    # Clean up
    import os
    if os.path.exists("test_debug2.db"):
        os.remove("test_debug2.db")
    return "Scraper initialized"

def test_metrics_analyzer():
    """Test metrics analyzer."""
    from database.models import DatabaseManager
    from analytics.metrics_analyzer import MetricsAnalyzer
    db = DatabaseManager("sqlite:///test_debug3.db")
    db.create_tables()
    analyzer = MetricsAnalyzer(db)
    db.close()
    # Clean up
    import os
    if os.path.exists("test_debug3.db"):
        os.remove("test_debug3.db")
    return "Metrics analyzer initialized"

def test_visualizer():
    """Test visualizer."""
    from ui.visualization import MetricsVisualizer
    viz = MetricsVisualizer()
    return "Visualizer initialized"

def test_optional_cache():
    """Test cache manager (optional)."""
    try:
        from database.cache_manager import CacheManager
        cache = CacheManager()
        return "Cache manager initialized"
    except ImportError:
        return "Cache manager not available (optional)"

def test_optional_sentiment():
    """Test sentiment analyzer (optional)."""
    try:
        from analytics.advanced_sentiment import AdvancedSentimentAnalyzer
        sentiment = AdvancedSentimentAnalyzer()
        return "Advanced sentiment analyzer initialized"
    except ImportError:
        return "Advanced sentiment analyzer not available (optional)"

def test_optional_validator():
    """Test data validator (optional)."""
    try:
        from analytics.data_validator import DataValidator
        validator = DataValidator()
        return "Data validator initialized"
    except ImportError:
        return "Data validator not available (optional)"

def test_optional_realtime():
    """Test real-time monitor (optional)."""
    try:
        from ui.realtime_monitor import RealTimeMonitor
        monitor = RealTimeMonitor()
        return "Real-time monitor initialized"
    except ImportError:
        return "Real-time monitor not available (optional)"

def test_optional_system_monitor():
    """Test system monitor (optional)."""
    try:
        from monitoring.system_monitor import get_system_monitor
        monitor = get_system_monitor()
        return "System monitor initialized"
    except ImportError:
        return "System monitor not available (optional)"

def test_settings():
    """Test settings loading."""
    try:
        from config.advanced_settings import get_settings
        settings = get_settings()
        return f"Advanced settings loaded - {settings.app_name}"
    except Exception as e:
        # Test fallback
        print(f"   Advanced settings failed: {e}")
        print("   Testing fallback minimal settings...")
        
        class MinimalSettings:
            app_name = "Test App"
            app_version = "1.0.0"
            host = "0.0.0.0"
            port = 7860
            debug = False
            
            def get_database_url(self):
                return "sqlite:///test.db"
            
            @property
            def features(self):
                return {'caching': False, 'data_validation': False, 'api_endpoints': False}
        
        settings = MinimalSettings()
        return f"Minimal settings created - {settings.app_name}"

def test_enhanced_tracker_init():
    """Test EnhancedRedditMentionTracker initialization."""
    from app import EnhancedRedditMentionTracker
    tracker = EnhancedRedditMentionTracker()
    return f"Enhanced tracker initialized - {tracker.settings.app_name}"

def test_gradio_interface_creation():
    """Test Gradio interface creation."""
    from app import EnhancedRedditMentionTracker
    tracker = EnhancedRedditMentionTracker()
    interface = tracker.create_gradio_interface()
    return "Gradio interface created successfully"

def test_gradio_launch():
    """Test Gradio interface launch (timeout after 10s)."""
    import threading
    import time
    
    from app import EnhancedRedditMentionTracker
    
    tracker = EnhancedRedditMentionTracker()
    interface = tracker.create_gradio_interface()
    
    # Launch in separate thread with timeout
    launch_result = {"success": False, "error": None}
    
    def launch_interface():
        try:
            interface.launch(
                server_port=7860,
                server_name="0.0.0.0",
                debug=False,
                share=False,
                show_error=True,
                quiet=False
            )
            launch_result["success"] = True
        except Exception as e:
            launch_result["error"] = str(e)
    
    launch_thread = threading.Thread(target=launch_interface, daemon=True)
    launch_thread.start()
    
    # Wait up to 10 seconds for launch
    for i in range(10):
        time.sleep(1)
        # Check if port is active
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', 7860))
                if result == 0:  # Port is in use
                    return "Gradio interface launched successfully"
        except:
            pass
        
        if launch_result["error"]:
            raise Exception(launch_result["error"])
    
    raise Exception("Gradio launch timed out after 10 seconds")

def main():
    """Run comprehensive startup debugging."""
    print("🔍 COMPREHENSIVE STARTUP DEBUG")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Settings Loading", test_settings),
        ("Database Manager", test_database_manager),
        ("Scraper", test_scraper),
        ("Metrics Analyzer", test_metrics_analyzer),
        ("Visualizer", test_visualizer),
        ("Cache Manager (Optional)", test_optional_cache),
        ("Sentiment Analyzer (Optional)", test_optional_sentiment),
        ("Data Validator (Optional)", test_optional_validator),
        ("Real-time Monitor (Optional)", test_optional_realtime),
        ("System Monitor (Optional)", test_optional_system_monitor),
        ("Enhanced Tracker Init", test_enhanced_tracker_init),
        ("Gradio Interface Creation", test_gradio_interface_creation),
        ("Gradio Launch Test", test_gradio_launch),
    ]
    
    results = []
    total_time = time.time()
    
    for name, test_func in tests:
        success, result = test_component(name, test_func)
        results.append((name, success, result))
        
        if not success and "Optional" not in name:
            print(f"\n❌ CRITICAL FAILURE: {name}")
            print("Stopping tests due to critical component failure.")
            break
    
    total_elapsed = time.time() - total_time
    
    print(f"\n📊 TEST RESULTS SUMMARY ({total_elapsed:.2f}s total)")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for name, success, result in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        print("🎉 All tests passed! System should start normally.")
    else:
        print(f"⚠️  {failed} tests failed. Check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(130) 