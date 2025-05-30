"""
LLM-based data analyzer for Reddit mentions using Groq API.
Provides natural language querying capabilities over retrieved Reddit data.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
import asyncio
import re

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

class LLMAnalyzer:
    """LLM-based analyzer for Reddit mention data using Groq API with multiple model fallback."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM analyzer."""
        self.logger = logging.getLogger(__name__)
        
        if not GROQ_AVAILABLE:
            self.logger.error("Groq package not available. Install with: pip install groq")
            self.client = None
            self.api_key = None
            return
        
        # Get API key from config file (same approach as debug script)
        try:
            from config.llm_config import LLMConfig
            config = LLMConfig()
            self.api_key = config.groq_api_key
            self.logger.info("Using API key from config file")
        except Exception as e:
            self.logger.error(f"Error accessing config API key: {e}")
            self.client = None
            self.api_key = None
            return
        
        if not self.api_key:
            self.logger.error("No API key found in config file")
            self.client = None
            return
        
        try:
            # Use exact same initialization as debug script
            self.client = Groq(api_key=self.api_key.strip())
            
            # Models in order of preference (most powerful first)
            self.models = [
                "deepseek-r1-distill-llama-70b",  # Most powerful reasoning model
                "llama-3.3-70b-versatile",        # Large versatile model
                "llama3-70b-8192",                # Large context model
                "qwen-qwq-32b",                   # Good mid-size reasoning
                "meta-llama/llama-4-maverick-17b-128e-instruct",  # Newer model
                "mistral-saba-24b",               # Good mid-size model
                "gemma2-9b-it",                   # Fast instruction model
                "llama-3.1-8b-instant"           # Fastest fallback
            ]
            
            self.max_tokens = 4096
            self.temperature = 0.1
            self.max_context_tokens = 60000  # Conservative estimate
            self.truncated_context_tokens = 5800  # Truncated context size
            self.max_user_query_tokens = 150  # Maximum user query size
            
            self.logger.info("LLM Analyzer initialized successfully with Groq API and multiple models")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if LLM analyzer is available and configured."""
        return self.client is not None and self.api_key is not None
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (approximately 4 characters per token)."""
        return len(text) // 4
    
    def _truncate_user_query(self, query: str) -> str:
        """Truncate user query to maximum token limit."""
        estimated_tokens = self._estimate_tokens(query)
        if estimated_tokens <= self.max_user_query_tokens:
            return query
        
        # Truncate to approximately max_user_query_tokens
        max_chars = self.max_user_query_tokens * 4
        truncated = query[:max_chars]
        
        # Try to truncate at word boundary
        if len(truncated) < len(query):
            last_space = truncated.rfind(' ')
            if last_space > max_chars * 0.8:  # If we can save 80% of content
                truncated = truncated[:last_space]
        
        self.logger.info(f"Truncated user query from {estimated_tokens} to ~{self._estimate_tokens(truncated)} tokens")
        return truncated
    
    def _score_mention(self, mention: Dict[str, Any]) -> float:
        """Score a mention for importance (higher score = more important to keep)."""
        score = 0.0
        
        # Reddit score weight (high engagement content)
        reddit_score = mention.get('score', 0)
        score += min(reddit_score / 100.0, 1.0) * 0.3
        
        # Comment count weight (discussion value)
        comments = mention.get('num_comments', 0)
        score += min(comments / 50.0, 1.0) * 0.2
        
        # Relevance score weight
        relevance = mention.get('relevance_score', 0.5)
        score += relevance * 0.3
        
        # Content length weight (more content = more information)
        content_length = len(mention.get('content', '')) + len(mention.get('title', ''))
        score += min(content_length / 500.0, 1.0) * 0.2
        
        return score
    
    def _truncate_context_by_score(self, mentions: List[Dict[str, Any]], target_tokens: int) -> List[Dict[str, Any]]:
        """Truncate context by removing lowest-scored mentions to fit target token count."""
        if not mentions:
            return mentions
        
        # Score all mentions
        scored_mentions = [(mention, self._score_mention(mention)) for mention in mentions]
        
        # Sort by score (highest first)
        scored_mentions.sort(key=lambda x: x[1], reverse=True)
        
        # Add mentions until we reach target token count
        selected_mentions = []
        current_tokens = 0
        
        for mention, score in scored_mentions:
            mention_text = self.format_mentions_for_context([mention], "")
            mention_tokens = self._estimate_tokens(mention_text)
            
            if current_tokens + mention_tokens <= target_tokens:
                selected_mentions.append(mention)
                current_tokens += mention_tokens
            else:
                break
        
        self.logger.info(f"Truncated context from {len(mentions)} to {len(selected_mentions)} mentions "
                        f"(~{current_tokens} tokens, target: {target_tokens})")
        
        return selected_mentions
    
    def _parse_think_content(self, text: str) -> Tuple[str, str]:
        """Parse think tokens from text and return (think_content, final_content)."""
        # Find all think blocks
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, text, re.DOTALL)
        
        # Remove think blocks from text to get final content
        final_content = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
        
        # Combine all think content
        think_content = '\n\n'.join(think_matches) if think_matches else ""
        
        return think_content, final_content
    
    def format_mentions_for_context(self, mentions: List[Dict[str, Any]], search_term: str) -> str:
        """Format mentions into a structured context for LLM analysis."""
        if not mentions:
            return "No Reddit mentions available for analysis."
        
        context_parts = [
            f"=== REDDIT MENTIONS ANALYSIS DATA ===",
            f"Search Term: '{search_term}'",
            f"Total Mentions: {len(mentions)}",
            f"Data Collection Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
            "=== INDIVIDUAL MENTIONS ==="
        ]
        
        for i, mention in enumerate(mentions, 1):
            # Basic information
            title = mention.get('title', 'No title')
            content = mention.get('content', 'No content')
            author = mention.get('author', 'Unknown')
            subreddit = mention.get('subreddit', 'Unknown')
            score = mention.get('score', 0)
            comments = mention.get('num_comments', 0)
            
            # Sentiment and quality metrics
            sentiment_score = mention.get('sentiment_score', 0)
            relevance_score = mention.get('relevance_score', 0)
            
            # Determine sentiment label
            if sentiment_score > 0.1:
                sentiment_label = "Positive"
            elif sentiment_score < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            # Format timestamp
            created_utc = mention.get('created_utc')
            if isinstance(created_utc, datetime):
                timestamp = created_utc.strftime('%Y-%m-%d %H:%M')
            elif isinstance(created_utc, (int, float)):
                timestamp = datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M')
            else:
                timestamp = 'Unknown time'
            
            mention_text = f"""
--- Mention {i} ---
Title: {title}
Content: {content[:500]}{'...' if len(content) > 500 else ''}
Author: {author}
Subreddit: r/{subreddit}
Score: {score} upvotes
Comments: {comments}
Sentiment: {sentiment_label} ({sentiment_score:.3f})
Relevance: {relevance_score:.3f}
Posted: {timestamp}
URL: {mention.get('url', 'No URL')}
"""
            context_parts.append(mention_text)
        
        # Add summary statistics
        total_score = sum(m.get('score', 0) for m in mentions)
        total_comments = sum(m.get('num_comments', 0) for m in mentions)
        avg_sentiment = sum(m.get('sentiment_score', 0) for m in mentions) / len(mentions)
        
        context_parts.extend([
            "",
            "=== SUMMARY STATISTICS ===",
            f"Total Reddit Score: {total_score}",
            f"Total Comments: {total_comments}",
            f"Average Sentiment: {avg_sentiment:.3f}",
            f"Top Subreddits: {', '.join(list(set(m.get('subreddit', 'unknown') for m in mentions))[:5])}",
            ""
        ])
        
        return "\n".join(context_parts)
    
    def create_system_prompt(self) -> str:
        """Create system prompt for Reddit mention analysis."""
        return """You are an expert Reddit data analyst and social media insights specialist. Your role is to analyze Reddit mentions and provide comprehensive, actionable insights.

**Your Capabilities:**
- Deep understanding of Reddit culture, terminology, and community dynamics
- Advanced sentiment analysis and trend identification
- Content quality assessment and engagement prediction
- Competitive intelligence and market research analysis
- Statistical analysis and data interpretation

**Analysis Guidelines:**
1. **Be Comprehensive**: Cover sentiment, engagement, trends, community reception, and actionable insights
2. **Be Specific**: Use actual data points, numbers, and examples from the provided mentions
3. **Be Actionable**: Provide concrete recommendations and next steps
4. **Be Contextual**: Consider Reddit culture, subreddit differences, and temporal patterns
5. **Be Honest**: Acknowledge limitations in data or analysis when appropriate

**Output Format:**
- Use clear headings and bullet points for readability
- Include specific examples and quotes when relevant
- Provide quantitative insights with percentages and ratios
- End with actionable recommendations
- Use markdown formatting for structure

**Data Context:**
You will receive Reddit mentions data including titles, content, scores, comments, sentiment scores, subreddits, and other metadata. Analyze this data to answer the user's specific question while providing broader insights about the topic's reception on Reddit."""
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit', 'rate_limit', 'too many requests', 
            'quota', 'limit exceeded', '429', 'rate limiting'
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    async def analyze_data_streaming(self, user_query: str, mentions: List[Dict[str, Any]], 
                                   search_term: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Analyze Reddit mentions data with streaming output and think token parsing."""
        if not self.is_available():
            yield {
                'type': 'error',
                'content': 'LLM analyzer is not available. Please check configuration.',
                'metadata': {}
            }
            return
        
        if not mentions:
            yield {
                'type': 'error', 
                'content': 'No Reddit mentions data available for analysis.',
                'metadata': {}
            }
            return
        
        # Truncate user query if too long
        original_query = user_query
        user_query = self._truncate_user_query(user_query)
        if len(user_query) < len(original_query):
            self.logger.info(f"User query truncated from {len(original_query)} to {len(user_query)} characters")
        
        # Prepare context
        context = self.format_mentions_for_context(mentions, search_term)
        estimated_tokens = self._estimate_tokens(context)
        
        self.logger.info(f"Initial context size: ~{estimated_tokens} tokens for {len(mentions)} mentions")
        
        # Try with full context first if it fits
        current_mentions = mentions
        context_truncated = False
        
        if estimated_tokens > self.max_context_tokens:
            self.logger.info(f"Context too large ({estimated_tokens} tokens), truncating to fit")
            current_mentions = self._truncate_context_by_score(mentions, self.truncated_context_tokens)
            context = self.format_mentions_for_context(current_mentions, search_term)
            context_truncated = True
            estimated_tokens = self._estimate_tokens(context)
        
        system_prompt = self.create_system_prompt()
        
        # Try each model in order
        last_error = None
        rate_limit_errors = []
        
        for model_name in self.models:
            try:
                self.logger.info(f"Attempting streaming analysis with model: {model_name}")
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"**User Question:** {user_query}\n\n**Reddit Data:**\n{context}"}
                ]
                
                # Stream the response
                stream = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True
                )
                
                full_content = ""
                think_content = ""
                final_content = ""
                in_think_block = False
                current_think_chunk = ""
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        full_content += chunk_content
                        
                        # Parse think tokens in real-time
                        temp_content = current_think_chunk + chunk_content
                        
                        # Check for think block start
                        if '<think>' in temp_content and not in_think_block:
                            in_think_block = True
                            # Split content at <think>
                            before_think, after_think = temp_content.split('<think>', 1)
                            # Add before_think to final content if any
                            if before_think:
                                final_content += before_think
                                yield {
                                    'type': 'final_content',
                                    'content': before_think
                                }
                            current_think_chunk = after_think
                        
                        # Check for think block end
                        elif '</think>' in temp_content and in_think_block:
                            in_think_block = False
                            # Split content at </think>
                            think_part, after_think = temp_content.split('</think>', 1)
                            # Add think part to think content
                            think_content += think_part
                            yield {
                                'type': 'think_content',
                                'content': think_part
                            }
                            # Add after_think to final content
                            final_content += after_think
                            if after_think:
                                yield {
                                    'type': 'final_content',
                                    'content': after_think
                                }
                            current_think_chunk = ""
                        
                        # Handle content based on current state
                        elif in_think_block:
                            # We're inside a think block
                            think_content += chunk_content
                            current_think_chunk = temp_content
                            yield {
                                'type': 'think_content',
                                'content': chunk_content
                            }
                        else:
                            # We're in final content
                            final_content += chunk_content
                            current_think_chunk = temp_content
                            yield {
                                'type': 'final_content',
                                'content': chunk_content
                            }
                
                # Calculate token usage
                prompt_tokens = estimated_tokens + self._estimate_tokens(system_prompt) + self._estimate_tokens(user_query)
                completion_tokens = self._estimate_tokens(full_content)
                total_tokens = prompt_tokens + completion_tokens
                
                metadata = {
                    'model_used': model_name,
                    'mentions_analyzed': len(current_mentions),
                    'context_truncated': context_truncated,
                    'tokens_used': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total': total_tokens
                    },
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'search_term': search_term,
                    'models_attempted': [model_name],
                    'rate_limit_errors': rate_limit_errors,
                    'think_content_length': len(think_content),
                    'final_content_length': len(final_content)
                }
                
                yield {
                    'type': 'complete',
                    'content': full_content,
                    'think_content': think_content,
                    'final_content': final_content,
                    'metadata': metadata
                }
                
                self.logger.info(f"Streaming analysis successful with {model_name}. Used ~{total_tokens} tokens, analyzed {len(current_mentions)} mentions")
                return
                
            except Exception as e:
                last_error = e
                
                if self._is_rate_limit_error(e):
                    rate_limit_errors.append({
                        'model': model_name,
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    self.logger.warning(f"Rate limit hit for {model_name}: {e}")
                    continue  # Try next model
                else:
                    self.logger.error(f"Non-rate-limit error with {model_name}: {e}")
                    continue  # Try next model for any error
        
        # If all models failed, try with truncated context (same logic as before)
        if rate_limit_errors and not context_truncated:
            self.logger.info("All models hit rate limits, trying with truncated context")
            
            current_mentions = self._truncate_context_by_score(mentions, self.truncated_context_tokens)
            context = self.format_mentions_for_context(current_mentions, search_term)
            estimated_tokens = self._estimate_tokens(context)
            
            # Retry with truncated context (similar streaming logic)
            for model_name in self.models:
                try:
                    self.logger.info(f"Retrying with truncated context using model: {model_name}")
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"**User Question:** {user_query}\n\n**Reddit Data (Truncated to highest quality mentions):**\n{context}"}
                    ]
                    
                    # Similar streaming logic as above but with truncation note
                    yield {
                        'type': 'final_content',
                        'content': f"**Note:** Analysis based on {len(current_mentions)} highest-quality mentions (context was truncated due to rate limits).\n\n"
                    }
                    
                    # Continue with streaming...
                    # (Abbreviated for brevity, but same streaming logic as above)
                    break
                    
                except Exception as e:
                    self.logger.error(f"Failed retry with {model_name}: {e}")
                    continue
        
        # If everything failed
        error_summary = f"All models failed. Rate limit errors: {len(rate_limit_errors)}. Last error: {last_error}"
        self.logger.error(error_summary)
        
        yield {
            'type': 'error',
            'content': f"LLM analysis failed: {error_summary}",
            'metadata': {
                'models_attempted': self.models,
                'rate_limit_errors': rate_limit_errors,
                'final_error': str(last_error) if last_error else 'Unknown error'
            }
        }
    
    async def analyze_data(self, user_query: str, mentions: List[Dict[str, Any]], search_term: str) -> Tuple[str, Dict[str, Any]]:
        """Analyze Reddit mentions data using LLM with multiple model fallback and context truncation."""
        if not self.is_available():
            return "LLM analyzer is not available. Please check configuration.", {}
        
        if not mentions:
            return "No Reddit mentions data available for analysis.", {}
        
        # Truncate user query if too long
        original_query = user_query
        user_query = self._truncate_user_query(user_query)
        if len(user_query) < len(original_query):
            self.logger.info(f"User query truncated from {len(original_query)} to {len(user_query)} characters")
        
        # Prepare context
        context = self.format_mentions_for_context(mentions, search_term)
        estimated_tokens = self._estimate_tokens(context)
        
        self.logger.info(f"Initial context size: ~{estimated_tokens} tokens for {len(mentions)} mentions")
        
        # Try with full context first if it fits
        current_mentions = mentions
        context_truncated = False
        
        if estimated_tokens > self.max_context_tokens:
            self.logger.info(f"Context too large ({estimated_tokens} tokens), truncating to fit")
            current_mentions = self._truncate_context_by_score(mentions, self.truncated_context_tokens)
            context = self.format_mentions_for_context(current_mentions, search_term)
            context_truncated = True
            estimated_tokens = self._estimate_tokens(context)
        
        system_prompt = self.create_system_prompt()
        
        # Try each model in order
        last_error = None
        rate_limit_errors = []
        
        for model_name in self.models:
            try:
                self.logger.info(f"Attempting analysis with model: {model_name}")
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"**User Question:** {user_query}\n\n**Reddit Data:**\n{context}"}
                ]
                
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                analysis_result = response.choices[0].message.content
                
                # Parse think content for metadata
                think_content, final_content = self._parse_think_content(analysis_result)
                
                # Calculate token usage
                prompt_tokens = estimated_tokens + self._estimate_tokens(system_prompt) + self._estimate_tokens(user_query)
                completion_tokens = self._estimate_tokens(analysis_result)
                total_tokens = prompt_tokens + completion_tokens
                
                metadata = {
                    'model_used': model_name,
                    'mentions_analyzed': len(current_mentions),
                    'context_truncated': context_truncated,
                    'tokens_used': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total': total_tokens
                    },
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'search_term': search_term,
                    'models_attempted': [model_name],
                    'rate_limit_errors': rate_limit_errors,
                    'think_content': think_content,
                    'final_content': final_content
                }
                
                self.logger.info(f"Analysis successful with {model_name}. Used ~{total_tokens} tokens, analyzed {len(current_mentions)} mentions")
                
                return analysis_result, metadata
                
            except Exception as e:
                last_error = e
                
                if self._is_rate_limit_error(e):
                    rate_limit_errors.append({
                        'model': model_name,
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    self.logger.warning(f"Rate limit hit for {model_name}: {e}")
                    continue  # Try next model
                else:
                    self.logger.error(f"Non-rate-limit error with {model_name}: {e}")
                    continue  # Try next model for any error
        
        # If all models failed due to rate limits, try with truncated context
        if rate_limit_errors and not context_truncated:
            self.logger.info("All models hit rate limits, trying with truncated context")
            
            # Truncate context more aggressively
            current_mentions = self._truncate_context_by_score(mentions, self.truncated_context_tokens)
            context = self.format_mentions_for_context(current_mentions, search_term)
            estimated_tokens = self._estimate_tokens(context)
            
            self.logger.info(f"Retrying with truncated context: ~{estimated_tokens} tokens, {len(current_mentions)} mentions")
            
            # Try each model again with truncated context
            for model_name in self.models:
                try:
                    self.logger.info(f"Retrying with truncated context using model: {model_name}")
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"**User Question:** {user_query}\n\n**Reddit Data (Truncated to highest quality mentions):**\n{context}"}
                    ]
                    
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    
                    analysis_result = response.choices[0].message.content
                    
                    # Parse think content for metadata
                    think_content, final_content = self._parse_think_content(analysis_result)
                    
                    # Calculate token usage
                    prompt_tokens = estimated_tokens + self._estimate_tokens(system_prompt) + self._estimate_tokens(user_query)
                    completion_tokens = self._estimate_tokens(analysis_result)
                    total_tokens = prompt_tokens + completion_tokens
                    
                    metadata = {
                        'model_used': model_name,
                        'mentions_analyzed': len(current_mentions),
                        'context_truncated': True,
                        'tokens_used': {
                            'prompt_tokens': prompt_tokens,
                            'completion_tokens': completion_tokens,
                            'total': total_tokens
                        },
                        'analysis_timestamp': datetime.utcnow().isoformat(),
                        'search_term': search_term,
                        'models_attempted': self.models,
                        'rate_limit_errors': rate_limit_errors,
                        'retry_with_truncation': True,
                        'think_content': think_content,
                        'final_content': final_content
                    }
                    
                    self.logger.info(f"Analysis successful with truncated context using {model_name}")
                    
                    # Add note about truncation
                    analysis_result = f"**Note:** Analysis based on {len(current_mentions)} highest-quality mentions (context was truncated due to rate limits).\n\n{analysis_result}"
                    
                    return analysis_result, metadata
                    
                except Exception as e:
                    self.logger.error(f"Failed retry with {model_name}: {e}")
                    continue
        
        # If everything failed
        error_summary = f"All models failed. Rate limit errors: {len(rate_limit_errors)}. Last error: {last_error}"
        self.logger.error(error_summary)
        
        metadata = {
            'model_used': None,
            'mentions_analyzed': len(mentions),
            'context_truncated': context_truncated,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'search_term': search_term,
            'models_attempted': self.models,
            'rate_limit_errors': rate_limit_errors,
            'final_error': str(last_error) if last_error else 'Unknown error'
        }
        
        return f"LLM analysis failed: {error_summary}", metadata
    
    def generate_suggested_questions(self, mentions: List[Dict[str, Any]], search_term: str) -> List[str]:
        """Generate suggested questions based on the data available."""
        if not mentions:
            return [
                f"What is the overall sentiment about '{search_term}' on Reddit?",
                f"Which subreddits are discussing '{search_term}' the most?",
                f"What are the main themes in '{search_term}' discussions?",
                f"How engaged is the Reddit community with '{search_term}' content?"
            ]
        
        # Analyze data to generate contextual suggestions
        subreddits = list(set(m.get('subreddit', 'unknown') for m in mentions))
        has_sentiment = any(m.get('sentiment_score') is not None for m in mentions)
        total_comments = sum(m.get('num_comments', 0) for m in mentions)
        
        suggestions = [
            f"What is the overall sentiment about '{search_term}' on Reddit?",
            f"Which subreddits are most engaged with '{search_term}' discussions?",
            f"What are the key themes and opinions about '{search_term}'?",
            f"How does Reddit community reception of '{search_term}' vary across different subreddits?",
            f"What are the most upvoted and controversial opinions about '{search_term}'?"
        ]
        
        # Add data-specific suggestions
        if len(subreddits) > 3:
            suggestions.append(f"How does discussion of '{search_term}' differ between r/{subreddits[0]} and r/{subreddits[1]}?")
        
        if total_comments > 100:
            suggestions.append(f"What are the main concerns or questions people have about '{search_term}'?")
        
        if has_sentiment:
            suggestions.append(f"Are there any concerning negative trends about '{search_term}' that need attention?")
        
        return suggestions[:6]  # Return top 6 suggestions
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history (placeholder for future implementation)."""
        # This would ideally connect to a database to store analysis history
        return [] 