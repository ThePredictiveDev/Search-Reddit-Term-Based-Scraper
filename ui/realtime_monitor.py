"""
Real-time monitoring and WebSocket support for live updates.
"""
import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
import threading
from queue import Queue
import uuid
import time

class RealTimeMonitor:
    """Real-time monitoring system with WebSocket support."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.session_clients: Dict[int, Set[WebSocketServerProtocol]] = {}
        self.message_queue = Queue()
        self.server = None
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
        # Event types
        self.EVENT_TYPES = {
            'SEARCH_STARTED': 'search_started',
            'SEARCH_PROGRESS': 'search_progress',
            'SEARCH_COMPLETED': 'search_completed',
            'SEARCH_FAILED': 'search_failed',
            'MENTION_FOUND': 'mention_found',
            'METRICS_UPDATED': 'metrics_updated',
            'CACHE_HIT': 'cache_hit',
            'SYSTEM_STATUS': 'system_status'
        }
    
    async def start_server(self):
        """Start the WebSocket server."""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port
            )
            self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Start message processing task
            asyncio.create_task(self.process_messages())
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {str(e)}")
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connections."""
        client_id = str(uuid.uuid4())
        self.clients.add(websocket)
        self.logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        try:
            # Send welcome message
            await self.send_to_client(websocket, {
                'type': 'connection_established',
                'client_id': client_id,
                'timestamp': datetime.utcnow().isoformat(),
                'message': 'Connected to Reddit Mention Tracker'
            })
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await self.send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    await self.send_error(websocket, f"Error processing message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {str(e)}")
        finally:
            self.clients.discard(websocket)
            # Remove from session-specific clients
            for session_id, session_clients in self.session_clients.items():
                session_clients.discard(websocket)
    
    async def handle_client_message(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle messages from clients."""
        message_type = data.get('type')
        
        if message_type == 'subscribe_session':
            session_id = data.get('session_id')
            if session_id:
                if session_id not in self.session_clients:
                    self.session_clients[session_id] = set()
                self.session_clients[session_id].add(websocket)
                
                await self.send_to_client(websocket, {
                    'type': 'subscription_confirmed',
                    'session_id': session_id,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        elif message_type == 'unsubscribe_session':
            session_id = data.get('session_id')
            if session_id and session_id in self.session_clients:
                self.session_clients[session_id].discard(websocket)
        
        elif message_type == 'ping':
            await self.send_to_client(websocket, {
                'type': 'pong',
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def send_to_client(self, websocket: WebSocketServerProtocol, data: Dict):
        """Send data to a specific client."""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)
        except Exception as e:
            self.logger.error(f"Error sending to client: {str(e)}")
    
    async def send_error(self, websocket: WebSocketServerProtocol, error_message: str):
        """Send error message to client."""
        await self.send_to_client(websocket, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def broadcast_to_all(self, data: Dict):
        """Broadcast data to all connected clients."""
        if not self.clients:
            return
        
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                await client.send(json.dumps(data))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {str(e)}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    async def broadcast_to_session(self, session_id: int, data: Dict):
        """Broadcast data to clients subscribed to a specific session."""
        if session_id not in self.session_clients:
            return
        
        session_clients = self.session_clients[session_id].copy()
        disconnected_clients = set()
        
        for client in session_clients:
            try:
                await client.send(json.dumps(data))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                self.logger.error(f"Error broadcasting to session client: {str(e)}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.session_clients[session_id] -= disconnected_clients
    
    def emit_search_started(self, session_id: int, search_term: str, max_pages: int):
        """Emit search started event."""
        self.message_queue.put({
            'type': self.EVENT_TYPES['SEARCH_STARTED'],
            'session_id': session_id,
            'search_term': search_term,
            'max_pages': max_pages,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def emit_search_progress(self, session_id: int, progress: float, message: str, mentions_found: int = 0):
        """Emit search progress event."""
        self.message_queue.put({
            'type': self.EVENT_TYPES['SEARCH_PROGRESS'],
            'session_id': session_id,
            'progress': progress,
            'message': message,
            'mentions_found': mentions_found,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def emit_search_completed(self, session_id: int, total_mentions: int, duration: float):
        """Emit search completed event."""
        self.message_queue.put({
            'type': self.EVENT_TYPES['SEARCH_COMPLETED'],
            'session_id': session_id,
            'total_mentions': total_mentions,
            'duration': duration,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def emit_search_failed(self, session_id: int, error_message: str):
        """Emit search failed event."""
        self.message_queue.put({
            'type': self.EVENT_TYPES['SEARCH_FAILED'],
            'session_id': session_id,
            'error_message': error_message,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def emit_mention_found(self, session_id: int, mention_data: Dict):
        """Emit mention found event."""
        # Sanitize mention data for real-time transmission
        sanitized_mention = {
            'reddit_id': mention_data.get('reddit_id'),
            'title': mention_data.get('title', '')[:100] + '...' if len(mention_data.get('title', '')) > 100 else mention_data.get('title', ''),
            'subreddit': mention_data.get('subreddit'),
            'score': mention_data.get('score', 0),
            'num_comments': mention_data.get('num_comments', 0),
            'url': mention_data.get('url'),
            'relevance_score': mention_data.get('relevance_score', 0.0)
        }
        
        self.message_queue.put({
            'type': self.EVENT_TYPES['MENTION_FOUND'],
            'session_id': session_id,
            'mention_data': mention_data,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def emit_metrics_updated(self, session_id: int, metrics_summary: Dict):
        """Emit metrics updated event."""
        self.message_queue.put({
            'type': self.EVENT_TYPES['METRICS_UPDATED'],
            'session_id': session_id,
            'metrics_summary': metrics_summary,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def emit_cache_hit(self, search_term: str, cached_count: int):
        """Emit cache hit event."""
        self.message_queue.put({
            'type': self.EVENT_TYPES['CACHE_HIT'],
            'search_term': search_term,
            'cached_count': cached_count,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def emit_system_status(self, status: Dict):
        """Emit system status event."""
        self.message_queue.put({
            'type': self.EVENT_TYPES['SYSTEM_STATUS'],
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def process_messages(self):
        """Process queued messages and broadcast them."""
        while True:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    
                    # Broadcast to all clients
                    await self.broadcast_to_all(message)
                    
                    # Also broadcast to session-specific clients if applicable
                    session_id = message.get('session_id')
                    if session_id:
                        await self.broadcast_to_session(session_id, message)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error processing messages: {str(e)}")
                await asyncio.sleep(1)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'total_clients': len(self.clients),
            'session_subscriptions': sum(len(clients) for clients in self.session_clients.values()),
            'server_running': self.server is not None,
            'server_status': 'running' if self.server is not None else 'stopped',
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            'queue_size': self.message_queue.qsize()
        }

class ProgressCallback:
    """Progress callback that integrates with real-time monitoring."""
    
    def __init__(self, monitor: RealTimeMonitor, session_id: int):
        self.monitor = monitor
        self.session_id = session_id
        self.mentions_found = 0
        self.start_time = datetime.utcnow()
    
    def __call__(self, message: str, progress: float = None, mention_data: Dict = None):
        """Called during scraping to report progress."""
        if mention_data:
            self.mentions_found += 1
            self.monitor.emit_mention_found(self.session_id, mention_data)
        
        if progress is not None:
            self.monitor.emit_search_progress(
                self.session_id, 
                progress, 
                message, 
                self.mentions_found
            )
        else:
            # Estimate progress based on message content
            estimated_progress = self._estimate_progress(message)
            self.monitor.emit_search_progress(
                self.session_id, 
                estimated_progress, 
                message, 
                self.mentions_found
            )
    
    def _estimate_progress(self, message: str) -> float:
        """Estimate progress based on message content."""
        message_lower = message.lower()
        
        if 'starting' in message_lower or 'initializing' in message_lower:
            return 0.1
        elif 'scraping page' in message_lower:
            # Try to extract page numbers
            import re
            match = re.search(r'page (\d+)/(\d+)', message_lower)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                return 0.2 + (0.6 * current / total)
            return 0.5
        elif 'saving' in message_lower:
            return 0.8
        elif 'generating' in message_lower or 'analytics' in message_lower:
            return 0.9
        elif 'complete' in message_lower:
            return 1.0
        else:
            return 0.5

# Global monitor instance
monitor = RealTimeMonitor()

def get_monitor() -> RealTimeMonitor:
    """Get the global monitor instance."""
    return monitor 