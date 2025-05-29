"""
Advanced data validation and quality assurance system for Reddit mentions.
"""
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics
import hashlib
from collections import defaultdict, Counter
import numpy as np
from textblob import TextBlob
import langdetect
from urllib.parse import urlparse

class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    quality_level: DataQuality
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityMetrics:
    """Quality metrics for a dataset."""
    total_records: int
    valid_records: int
    duplicate_records: int
    spam_records: int
    low_quality_records: int
    average_quality_score: float
    quality_distribution: Dict[DataQuality, int]
    common_issues: List[Tuple[str, int]]

class DataValidator:
    """Comprehensive data validation and quality assurance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation rules configuration
        self.validation_rules = {
            'title': {
                'min_length': 5,
                'max_length': 300,
                'required': True,
                'patterns': {
                    'spam_indicators': [
                        r'\b(click here|buy now|limited time|act now)\b',
                        r'\$\d+.*\b(free|discount|sale)\b',
                        r'\b(viagra|casino|lottery|winner)\b'
                    ],
                    'quality_indicators': [
                        r'\b(analysis|review|discussion|opinion|experience)\b',
                        r'\b(how to|guide|tutorial|tips)\b',
                        r'\b(comparison|vs|versus)\b'
                    ]
                }
            },
            'content': {
                'min_length': 10,
                'max_length': 10000,
                'required': False,
                'patterns': {
                    'spam_indicators': [
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        r'\b(upvote|subscribe|follow me|check out my)\b',
                        r'\b(affiliate|referral|promo code)\b'
                    ]
                }
            },
            'author': {
                'min_length': 1,
                'max_length': 50,
                'required': True,
                'patterns': {
                    'suspicious': [
                        r'^[a-zA-Z]+\d+$',  # username followed by numbers
                        r'^(bot|spam|fake)',
                        r'\d{4,}'  # long number sequences
                    ]
                }
            },
            'subreddit': {
                'min_length': 1,
                'max_length': 50,
                'required': True,
                'whitelist': [
                    'technology', 'programming', 'artificial', 'MachineLearning',
                    'business', 'entrepreneur', 'startups', 'investing',
                    'news', 'worldnews', 'politics', 'economics',
                    'cryptocurrency', 'bitcoin', 'ethereum'
                ]
            },
            'score': {
                'min_value': -1000,
                'max_value': 100000,
                'required': True
            },
            'created_utc': {
                'required': True,
                'max_age_days': 30
            }
        }
        
        # Language detection
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
        # Spam detection patterns
        self.spam_patterns = {
            'promotional': [
                r'\b(buy|purchase|order|shop)\b.*\b(now|today|here)\b',
                r'\b(discount|sale|offer|deal)\b.*\d+%',
                r'\b(free|bonus|gift)\b.*\b(limited|exclusive)\b'
            ],
            'suspicious_links': [
                r'bit\.ly|tinyurl|t\.co|goo\.gl',
                r'[a-z0-9]{8,}\.com',  # Random domain names
                r'click\..*\.com'
            ],
            'repetitive': [
                r'(.{10,})\1{2,}',  # Repeated text patterns
                r'\b(\w+)\s+\1\s+\1\b'  # Repeated words
            ]
        }
        
        # Quality scoring weights
        self.quality_weights = {
            'length_score': 0.15,
            'language_score': 0.10,
            'spam_score': 0.25,
            'engagement_score': 0.20,
            'content_quality': 0.15,
            'metadata_completeness': 0.10,
            'freshness_score': 0.05
        }
        
        # Cache for expensive operations
        self.validation_cache = {}
        self.language_cache = {}
    
    def validate_mention(self, mention: Dict[str, Any]) -> ValidationResult:
        """Validate a single Reddit mention."""
        # Generate cache key
        cache_key = self._generate_cache_key(mention)
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        issues = []
        suggestions = []
        quality_scores = {}
        
        # Basic field validation
        field_validation = self._validate_fields(mention)
        issues.extend(field_validation['issues'])
        suggestions.extend(field_validation['suggestions'])
        
        # Content quality analysis
        content_quality = self._analyze_content_quality(mention)
        quality_scores.update(content_quality['scores'])
        issues.extend(content_quality['issues'])
        suggestions.extend(content_quality['suggestions'])
        
        # Spam detection
        spam_analysis = self._detect_spam(mention)
        quality_scores['spam_score'] = spam_analysis['score']
        issues.extend(spam_analysis['issues'])
        
        # Language detection
        language_analysis = self._analyze_language(mention)
        quality_scores['language_score'] = language_analysis['score']
        if language_analysis['issues']:
            issues.extend(language_analysis['issues'])
        
        # Engagement analysis
        engagement_analysis = self._analyze_engagement(mention)
        quality_scores['engagement_score'] = engagement_analysis['score']
        
        # Metadata completeness
        metadata_analysis = self._analyze_metadata_completeness(mention)
        quality_scores['metadata_completeness'] = metadata_analysis['score']
        
        # Freshness analysis
        freshness_analysis = self._analyze_freshness(mention)
        quality_scores['freshness_score'] = freshness_analysis['score']
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality(quality_scores)
        quality_level = self._determine_quality_level(overall_score)
        
        # Determine if valid
        is_valid = self._determine_validity(issues, overall_score)
        
        result = ValidationResult(
            is_valid=is_valid,
            quality_score=overall_score,
            quality_level=quality_level,
            issues=issues,
            suggestions=suggestions,
            metadata={
                'individual_scores': quality_scores,
                'validation_timestamp': datetime.utcnow().isoformat()
            }
        )
        
        # Cache result
        self.validation_cache[cache_key] = result
        
        return result
    
    def validate_dataset(self, mentions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], QualityMetrics]:
        """Validate an entire dataset of mentions."""
        validated_mentions = []
        validation_results = []
        
        self.logger.info(f"Validating dataset of {len(mentions)} mentions")
        
        # Validate individual mentions
        for mention in mentions:
            result = self.validate_mention(mention)
            validation_results.append(result)
            
            if result.is_valid:
                # Don't add internal quality metadata to mention data
                # These fields are not part of the RedditMention model
                validated_mentions.append(mention)
        
        # Detect duplicates
        duplicates = self._detect_duplicates(validated_mentions)
        
        # Remove duplicates
        unique_mentions = self._remove_duplicates(validated_mentions, duplicates)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            mentions, validation_results, duplicates
        )
        
        self.logger.info(f"Validation complete: {len(unique_mentions)}/{len(mentions)} mentions passed")
        
        return unique_mentions, quality_metrics
    
    def _validate_fields(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual fields against rules."""
        issues = []
        suggestions = []
        
        for field, rules in self.validation_rules.items():
            value = mention.get(field)
            
            # Required field check
            if rules.get('required', False) and not value:
                issues.append({
                    'type': 'missing_required_field',
                    'field': field,
                    'level': ValidationLevel.ERROR,
                    'message': f"Required field '{field}' is missing"
                })
                continue
            
            if value is None:
                continue
            
            # String field validations
            if isinstance(value, str):
                # Length validations
                if 'min_length' in rules and len(value) < rules['min_length']:
                    issues.append({
                        'type': 'field_too_short',
                        'field': field,
                        'level': ValidationLevel.WARNING,
                        'message': f"Field '{field}' is too short ({len(value)} < {rules['min_length']})"
                    })
                
                if 'max_length' in rules and len(value) > rules['max_length']:
                    issues.append({
                        'type': 'field_too_long',
                        'field': field,
                        'level': ValidationLevel.WARNING,
                        'message': f"Field '{field}' is too long ({len(value)} > {rules['max_length']})"
                    })
                    suggestions.append(f"Consider truncating '{field}' to {rules['max_length']} characters")
                
                # Pattern validations
                if 'patterns' in rules:
                    for pattern_type, patterns in rules['patterns'].items():
                        for pattern in patterns:
                            if re.search(pattern, value, re.IGNORECASE):
                                level = ValidationLevel.WARNING if pattern_type == 'spam_indicators' else ValidationLevel.INFO
                                issues.append({
                                    'type': f'pattern_match_{pattern_type}',
                                    'field': field,
                                    'level': level,
                                    'message': f"Field '{field}' matches {pattern_type} pattern"
                                })
                
                # Whitelist check
                if 'whitelist' in rules and value.lower() not in [w.lower() for w in rules['whitelist']]:
                    issues.append({
                        'type': 'not_in_whitelist',
                        'field': field,
                        'level': ValidationLevel.INFO,
                        'message': f"Field '{field}' value '{value}' not in approved list"
                    })
            
            # Numeric field validations
            elif isinstance(value, (int, float)):
                if 'min_value' in rules and value < rules['min_value']:
                    issues.append({
                        'type': 'value_too_low',
                        'field': field,
                        'level': ValidationLevel.WARNING,
                        'message': f"Field '{field}' value {value} is below minimum {rules['min_value']}"
                    })
                
                if 'max_value' in rules and value > rules['max_value']:
                    issues.append({
                        'type': 'value_too_high',
                        'field': field,
                        'level': ValidationLevel.WARNING,
                        'message': f"Field '{field}' value {value} exceeds maximum {rules['max_value']}"
                    })
            
            # Date field validations
            elif field == 'created_utc':
                try:
                    if isinstance(value, str):
                        created_date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    else:
                        created_date = datetime.fromtimestamp(value)
                    
                    age_days = (datetime.utcnow() - created_date).days
                    max_age = rules.get('max_age_days', 365)
                    
                    if age_days > max_age:
                        issues.append({
                            'type': 'data_too_old',
                            'field': field,
                            'level': ValidationLevel.INFO,
                            'message': f"Data is {age_days} days old (max: {max_age})"
                        })
                except (ValueError, TypeError):
                    issues.append({
                        'type': 'invalid_date_format',
                        'field': field,
                        'level': ValidationLevel.ERROR,
                        'message': f"Invalid date format in field '{field}'"
                    })
        
        return {'issues': issues, 'suggestions': suggestions}
    
    def _analyze_content_quality(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content quality using various metrics."""
        issues = []
        suggestions = []
        scores = {}
        
        title = mention.get('title', '')
        content = mention.get('content', '')
        combined_text = f"{title} {content}".strip()
        
        # Length score
        length_score = self._calculate_length_score(combined_text)
        scores['length_score'] = length_score
        
        if length_score < 0.3:
            issues.append({
                'type': 'content_too_short',
                'level': ValidationLevel.WARNING,
                'message': "Content appears to be too short for meaningful analysis"
            })
            suggestions.append("Consider looking for posts with more substantial content")
        
        # Content quality score
        content_quality_score = self._calculate_content_quality_score(combined_text)
        scores['content_quality'] = content_quality_score
        
        if content_quality_score < 0.4:
            issues.append({
                'type': 'low_content_quality',
                'level': ValidationLevel.INFO,
                'message': "Content quality appears to be below average"
            })
        
        # Readability analysis
        readability_score = self._analyze_readability(combined_text)
        scores['readability_score'] = readability_score
        
        return {
            'scores': scores,
            'issues': issues,
            'suggestions': suggestions
        }
    
    def _detect_spam(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """Detect spam content using pattern matching and heuristics."""
        issues = []
        spam_indicators = 0
        total_checks = 0
        
        title = mention.get('title', '')
        content = mention.get('content', '')
        author = mention.get('author', '')
        url = mention.get('url', '')
        
        # Check spam patterns
        for category, patterns in self.spam_patterns.items():
            for pattern in patterns:
                total_checks += 1
                if re.search(pattern, f"{title} {content}", re.IGNORECASE):
                    spam_indicators += 1
                    issues.append({
                        'type': f'spam_pattern_{category}',
                        'level': ValidationLevel.WARNING,
                        'message': f"Content matches {category} spam pattern"
                    })
        
        # Check for suspicious URLs
        if url:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                for pattern in self.spam_patterns['suspicious_links']:
                    if re.search(pattern, parsed_url.netloc):
                        spam_indicators += 1
                        issues.append({
                            'type': 'suspicious_url',
                            'level': ValidationLevel.WARNING,
                            'message': f"URL appears suspicious: {parsed_url.netloc}"
                        })
        
        # Check author patterns
        for pattern in self.validation_rules['author']['patterns']['suspicious']:
            if re.search(pattern, author, re.IGNORECASE):
                spam_indicators += 1
                issues.append({
                    'type': 'suspicious_author',
                    'level': ValidationLevel.INFO,
                    'message': f"Author name matches suspicious pattern"
                })
        
        # Calculate spam score (inverted - lower is better)
        spam_score = max(0, 1 - (spam_indicators / max(total_checks, 1)))
        
        return {
            'score': spam_score,
            'issues': issues,
            'spam_indicators': spam_indicators
        }
    
    def _analyze_language(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze language and detect non-English content."""
        issues = []
        
        title = mention.get('title', '')
        content = mention.get('content', '')
        combined_text = f"{title} {content}".strip()
        
        if not combined_text:
            return {'score': 0.0, 'issues': issues}
        
        # Check cache first
        text_hash = hashlib.md5(combined_text.encode()).hexdigest()
        if text_hash in self.language_cache:
            return self.language_cache[text_hash]
        
        try:
            detected_lang = langdetect.detect(combined_text)
            confidence = langdetect.detect_langs(combined_text)[0].prob
            
            if detected_lang in self.supported_languages:
                language_score = confidence
            else:
                language_score = 0.0
                issues.append({
                    'type': 'unsupported_language',
                    'level': ValidationLevel.INFO,
                    'message': f"Detected language '{detected_lang}' is not supported"
                })
            
            if confidence < 0.7:
                issues.append({
                    'type': 'low_language_confidence',
                    'level': ValidationLevel.WARNING,
                    'message': f"Language detection confidence is low ({confidence:.2f})"
                })
        
        except Exception:
            language_score = 0.5  # Neutral score if detection fails
            issues.append({
                'type': 'language_detection_failed',
                'level': ValidationLevel.WARNING,
                'message': "Could not detect language"
            })
        
        result = {'score': language_score, 'issues': issues}
        self.language_cache[text_hash] = result
        
        return result
    
    def _analyze_engagement(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement metrics."""
        score = mention.get('score', 0)
        num_comments = mention.get('num_comments', 0)
        
        # Normalize engagement score
        engagement_score = min(1.0, (score + num_comments * 2) / 100)
        
        return {'score': engagement_score}
    
    def _analyze_metadata_completeness(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze completeness of metadata."""
        required_fields = ['title', 'author', 'subreddit', 'created_utc', 'score']
        optional_fields = ['content', 'url', 'num_comments', 'upvote_ratio']
        
        present_required = sum(1 for field in required_fields if mention.get(field) is not None)
        present_optional = sum(1 for field in optional_fields if mention.get(field) is not None)
        
        completeness_score = (present_required / len(required_fields)) * 0.8 + \
                           (present_optional / len(optional_fields)) * 0.2
        
        return {'score': completeness_score}
    
    def _analyze_freshness(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data freshness."""
        created_utc = mention.get('created_utc')
        if not created_utc:
            return {'score': 0.0}
        
        try:
            if isinstance(created_utc, str):
                created_date = datetime.fromisoformat(created_utc.replace('Z', '+00:00'))
            else:
                created_date = datetime.fromtimestamp(created_utc)
            
            age_days = (datetime.utcnow() - created_date).days
            
            # Fresher data gets higher score
            if age_days <= 1:
                freshness_score = 1.0
            elif age_days <= 7:
                freshness_score = 0.8
            elif age_days <= 30:
                freshness_score = 0.6
            else:
                freshness_score = max(0.0, 1.0 - (age_days / 365))
            
            return {'score': freshness_score}
        
        except (ValueError, TypeError):
            return {'score': 0.0}
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length."""
        length = len(text.strip())
        
        if length < 20:
            return 0.1
        elif length < 50:
            return 0.3
        elif length < 100:
            return 0.6
        elif length < 500:
            return 0.8
        elif length < 2000:
            return 1.0
        else:
            return 0.9  # Very long text might be spam
    
    def _calculate_content_quality_score(self, text: str) -> float:
        """Calculate content quality score using various heuristics."""
        if not text.strip():
            return 0.0
        
        score = 0.5  # Base score
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        if 5 <= avg_sentence_length <= 25:
            score += 0.2
        
        # Vocabulary diversity
        words = re.findall(r'\b\w+\b', text.lower())
        if words:
            unique_words = len(set(words))
            diversity_ratio = unique_words / len(words)
            score += min(0.2, diversity_ratio * 0.4)
        
        # Punctuation usage
        punctuation_ratio = len(re.findall(r'[.!?,:;]', text)) / max(len(text), 1)
        if 0.02 <= punctuation_ratio <= 0.15:
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_readability(self, text: str) -> float:
        """Analyze text readability."""
        if not text.strip():
            return 0.0
        
        # Simple readability metrics
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len([s for s in sentences if s.strip()])
        avg_syllables_per_word = np.mean([self._count_syllables(word) for word in words])
        
        # Simplified Flesch Reading Ease approximation
        readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, readability / 100))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_overall_quality(self, scores: Dict[str, float]) -> float:
        """Calculate overall quality score using weighted average."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.quality_weights.items():
            if metric in scores:
                total_score += scores[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> DataQuality:
        """Determine quality level based on score."""
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.FAIR
        elif score >= 0.3:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID
    
    def _determine_validity(self, issues: List[Dict[str, Any]], quality_score: float) -> bool:
        """Determine if mention is valid based on issues and quality."""
        # Check for critical issues
        critical_issues = [issue for issue in issues if issue.get('level') == ValidationLevel.CRITICAL]
        if critical_issues:
            return False
        
        # Check for too many errors
        error_issues = [issue for issue in issues if issue.get('level') == ValidationLevel.ERROR]
        if len(error_issues) > 2:
            return False
        
        # Check minimum quality threshold
        if quality_score < 0.2:
            return False
        
        return True
    
    def _detect_duplicates(self, mentions: List[Dict[str, Any]]) -> Set[int]:
        """Detect duplicate mentions."""
        seen_hashes = set()
        duplicates = set()
        
        for i, mention in enumerate(mentions):
            # Create hash from key fields
            key_fields = [
                mention.get('title', ''),
                mention.get('author', ''),
                mention.get('subreddit', ''),
                str(mention.get('created_utc', ''))
            ]
            
            content_hash = hashlib.md5('|'.join(key_fields).encode()).hexdigest()
            
            if content_hash in seen_hashes:
                duplicates.add(i)
            else:
                seen_hashes.add(content_hash)
        
        return duplicates
    
    def _remove_duplicates(self, mentions: List[Dict[str, Any]], duplicates: Set[int]) -> List[Dict[str, Any]]:
        """Remove duplicate mentions."""
        return [mention for i, mention in enumerate(mentions) if i not in duplicates]
    
    def _calculate_quality_metrics(
        self, 
        original_mentions: List[Dict[str, Any]], 
        validation_results: List[ValidationResult],
        duplicates: Set[int]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        total_records = len(original_mentions)
        valid_records = sum(1 for result in validation_results if result.is_valid)
        duplicate_records = len(duplicates)
        
        # Count spam and low quality records
        spam_records = 0
        low_quality_records = 0
        quality_scores = []
        
        for result in validation_results:
            quality_scores.append(result.quality_score)
            
            if any(issue.get('type', '').startswith('spam_') for issue in result.issues):
                spam_records += 1
            
            if result.quality_level in [DataQuality.POOR, DataQuality.INVALID]:
                low_quality_records += 1
        
        # Quality distribution
        quality_distribution = Counter(result.quality_level for result in validation_results)
        
        # Common issues
        all_issues = []
        for result in validation_results:
            all_issues.extend([issue.get('type', 'unknown') for issue in result.issues])
        
        common_issues = Counter(all_issues).most_common(10)
        
        return QualityMetrics(
            total_records=total_records,
            valid_records=valid_records,
            duplicate_records=duplicate_records,
            spam_records=spam_records,
            low_quality_records=low_quality_records,
            average_quality_score=np.mean(quality_scores) if quality_scores else 0.0,
            quality_distribution=dict(quality_distribution),
            common_issues=common_issues
        )
    
    def _generate_cache_key(self, mention: Dict[str, Any]) -> str:
        """Generate cache key for validation result."""
        key_fields = [
            mention.get('title', ''),
            mention.get('content', ''),
            mention.get('author', ''),
            str(mention.get('score', 0))
        ]
        return hashlib.md5('|'.join(key_fields).encode()).hexdigest()
    
    def get_validation_summary(self, quality_metrics: QualityMetrics) -> str:
        """Generate human-readable validation summary."""
        summary = f"""
Data Validation Summary:
========================
Total Records: {quality_metrics.total_records}
Valid Records: {quality_metrics.valid_records} ({quality_metrics.valid_records/quality_metrics.total_records*100:.1f}%)
Duplicate Records: {quality_metrics.duplicate_records}
Spam Records: {quality_metrics.spam_records}
Low Quality Records: {quality_metrics.low_quality_records}
Average Quality Score: {quality_metrics.average_quality_score:.3f}

Quality Distribution:
"""
        
        for quality_level, count in quality_metrics.quality_distribution.items():
            percentage = count / quality_metrics.total_records * 100
            summary += f"  {quality_level.value.title()}: {count} ({percentage:.1f}%)\n"
        
        summary += "\nMost Common Issues:\n"
        for issue_type, count in quality_metrics.common_issues[:5]:
            summary += f"  {issue_type}: {count}\n"
        
        return summary 