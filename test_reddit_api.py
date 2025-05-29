#!/usr/bin/env python3
"""
Test script to verify Reddit API functionality
"""
import asyncio
import requests
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

class TestRedditAPI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RedditMentionTracker/1.0 (Educational Purpose)'
        })
    
    def test_reddit_json_api(self, query: str = "OpenAI"):
        """Test Reddit's JSON API directly."""
        print(f"Testing Reddit API search for: {query}")
        
        try:
            url = "https://www.reddit.com/search.json"
            params = {
                'q': query,
                'sort': 'relevance',
                't': 'week',
                'limit': 10,
                'type': 'link'
            }
            
            print(f"Making request to: {url}")
            print(f"Parameters: {params}")
            
            response = self.session.get(url, params=params, timeout=10)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response keys: {list(data.keys())}")
                
                if 'data' in data and 'children' in data['data']:
                    posts = data['data']['children']
                    print(f"Found {len(posts)} posts")
                    
                    for i, post in enumerate(posts[:3]):  # Show first 3
                        post_data = post['data']
                        print(f"\nPost {i+1}:")
                        print(f"  Title: {post_data.get('title', 'N/A')[:100]}...")
                        print(f"  Subreddit: {post_data.get('subreddit', 'N/A')}")
                        print(f"  Score: {post_data.get('score', 0)}")
                        print(f"  Comments: {post_data.get('num_comments', 0)}")
                        print(f"  URL: {post_data.get('permalink', 'N/A')}")
                    
                    return True
                else:
                    print("No data/children in response")
                    print(f"Full response: {json.dumps(data, indent=2)[:500]}...")
                    return False
            else:
                print(f"Bad response: {response.status_code}")
                print(f"Response text: {response.text[:500]}...")
                return False
                
        except Exception as e:
            print(f"API test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("=== Reddit API Test ===")
    
    tester = TestRedditAPI()
    
    # Test different queries
    test_queries = ["OpenAI", "ChatGPT", "artificial intelligence"]
    
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        success = tester.test_reddit_json_api(query)
        if success:
            print(f"✅ SUCCESS: API worked for '{query}'")
        else:
            print(f"❌ FAILED: API failed for '{query}'")
        
        # Small delay between requests
        import time
        time.sleep(2)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 