"""
Advanced sentiment analysis using transformer models.
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from textblob import TextBlob

# Optional imports for advanced features
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    pipeline = None

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis using BERT-based models."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available, falling back to TextBlob only")
            return
        
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize emotion analysis pipeline
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info(f"Advanced sentiment analyzer initialized with {model_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize transformer models: {str(e)}")
            self.sentiment_pipeline = None
            self.emotion_pipeline = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment with multiple approaches.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and confidence
        """
        if not text or not text.strip():
            return self._empty_sentiment()
        
        results = {}
        
        # TextBlob baseline (always available)
        try:
            blob = TextBlob(text)
            results['textblob_polarity'] = blob.sentiment.polarity
            results['textblob_subjectivity'] = blob.sentiment.subjectivity
        except Exception as e:
            self.logger.debug(f"TextBlob analysis failed: {str(e)}")
            results['textblob_polarity'] = 0.0
            results['textblob_subjectivity'] = 0.0
        
        # Advanced transformer-based analysis
        if self.sentiment_pipeline:
            try:
                # Truncate text to model's max length
                truncated_text = text[:512]
                
                sentiment_result = self.sentiment_pipeline(truncated_text)[0]
                
                # Convert to standardized scale (-1 to 1)
                label = sentiment_result['label'].upper()
                confidence = sentiment_result['score']
                
                if 'POSITIVE' in label:
                    results['transformer_sentiment'] = confidence
                elif 'NEGATIVE' in label:
                    results['transformer_sentiment'] = -confidence
                else:  # NEUTRAL
                    results['transformer_sentiment'] = 0.0
                
                results['transformer_confidence'] = confidence
                
            except Exception as e:
                self.logger.debug(f"Transformer sentiment analysis failed: {str(e)}")
                results['transformer_sentiment'] = 0.0
                results['transformer_confidence'] = 0.0
        
        # Emotion analysis
        if self.emotion_pipeline:
            try:
                emotion_result = self.emotion_pipeline(text[:512])[0]
                results['primary_emotion'] = emotion_result['label']
                results['emotion_confidence'] = emotion_result['score']
            except Exception as e:
                self.logger.debug(f"Emotion analysis failed: {str(e)}")
                results['primary_emotion'] = 'neutral'
                results['emotion_confidence'] = 0.0
        
        # Calculate composite sentiment score
        results['composite_sentiment'] = self._calculate_composite_sentiment(results)
        
        return results
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts efficiently."""
        if not texts:
            return []
        
        results = []
        
        # Batch process with transformers if available
        if self.sentiment_pipeline:
            try:
                # Truncate texts and batch process
                truncated_texts = [text[:512] for text in texts]
                
                sentiment_results = self.sentiment_pipeline(truncated_texts)
                
                for i, (text, sentiment_result) in enumerate(zip(texts, sentiment_results)):
                    result = self._process_single_result(text, sentiment_result)
                    results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Batch sentiment analysis failed: {str(e)}")
                # Fallback to individual processing
                for text in texts:
                    results.append(self.analyze_sentiment(text))
        else:
            # Process individually with TextBlob
            for text in texts:
                results.append(self.analyze_sentiment(text))
        
        return results
    
    def get_sentiment_insights(self, sentiments: List[Dict[str, float]]) -> Dict[str, any]:
        """Generate insights from a collection of sentiment analyses."""
        if not sentiments:
            return {}
        
        # Extract composite scores
        composite_scores = [s.get('composite_sentiment', 0.0) for s in sentiments]
        emotions = [s.get('primary_emotion', 'neutral') for s in sentiments]
        
        # Calculate statistics
        insights = {
            'total_analyzed': len(sentiments),
            'average_sentiment': np.mean(composite_scores),
            'sentiment_std': np.std(composite_scores),
            'positive_ratio': len([s for s in composite_scores if s > 0.1]) / len(composite_scores),
            'negative_ratio': len([s for s in composite_scores if s < -0.1]) / len(composite_scores),
            'neutral_ratio': len([s for s in composite_scores if -0.1 <= s <= 0.1]) / len(composite_scores),
        }
        
        # Emotion distribution
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        insights['emotion_distribution'] = emotion_counts
        insights['dominant_emotion'] = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Sentiment trend (if temporal data available)
        insights['sentiment_volatility'] = np.std(composite_scores)
        
        return insights
    
    def _process_single_result(self, text: str, sentiment_result: Dict) -> Dict[str, float]:
        """Process a single sentiment result."""
        result = {}
        
        # TextBlob analysis
        try:
            blob = TextBlob(text)
            result['textblob_polarity'] = blob.sentiment.polarity
            result['textblob_subjectivity'] = blob.sentiment.subjectivity
        except:
            result['textblob_polarity'] = 0.0
            result['textblob_subjectivity'] = 0.0
        
        # Transformer result
        label = sentiment_result['label'].upper()
        confidence = sentiment_result['score']
        
        if 'POSITIVE' in label:
            result['transformer_sentiment'] = confidence
        elif 'NEGATIVE' in label:
            result['transformer_sentiment'] = -confidence
        else:
            result['transformer_sentiment'] = 0.0
        
        result['transformer_confidence'] = confidence
        
        # Composite score
        result['composite_sentiment'] = self._calculate_composite_sentiment(result)
        
        return result
    
    def _calculate_composite_sentiment(self, results: Dict[str, float]) -> float:
        """Calculate a composite sentiment score from multiple analyses."""
        textblob_score = results.get('textblob_polarity', 0.0)
        transformer_score = results.get('transformer_sentiment', 0.0)
        transformer_confidence = results.get('transformer_confidence', 0.0)
        
        # Weight transformer result by confidence, fallback to TextBlob
        if transformer_confidence > 0.5:
            # High confidence transformer result
            composite = 0.7 * transformer_score + 0.3 * textblob_score
        elif transformer_confidence > 0.3:
            # Medium confidence - equal weight
            composite = 0.5 * transformer_score + 0.5 * textblob_score
        else:
            # Low confidence - prefer TextBlob
            composite = 0.3 * transformer_score + 0.7 * textblob_score
        
        # Ensure result is in [-1, 1] range
        return max(-1.0, min(1.0, composite))
    
    def _empty_sentiment(self) -> Dict[str, float]:
        """Return empty sentiment result."""
        return {
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'transformer_sentiment': 0.0,
            'transformer_confidence': 0.0,
            'primary_emotion': 'neutral',
            'emotion_confidence': 0.0,
            'composite_sentiment': 0.0
        }
    
    def is_available(self) -> bool:
        """Check if advanced sentiment analysis is available."""
        return self.sentiment_pipeline is not None 