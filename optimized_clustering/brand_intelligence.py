"""
Brand Intelligence Module

Provides intelligent brand analysis for logo clustering.
Classifies logos by brand family and industry.
"""

from collections import defaultdict


class BrandIntelligence:
    """Intelligent brand analysis for logo clustering"""
    
    def __init__(self):
        # Industry keyword mapping for brand classification
        self.industry_keywords = {
            'technology': ['tech', 'software', 'app', 'digital', 'cloud', 'ai', 'data', 
                          'microsoft', 'google', 'apple', 'amazon', 'meta', 'adobe',
                          'oracle', 'salesforce', 'zoom', 'slack', 'dropbox', 'github'],
            'finance': ['bank', 'financial', 'capital', 'investment', 'credit', 'loan',
                       'visa', 'mastercard', 'paypal', 'stripe', 'square', 'amex',
                       'jpmorgan', 'goldman', 'wells', 'chase', 'citibank'],
            'ecommerce': ['shop', 'store', 'market', 'buy', 'sell', 'retail', 'commerce',
                         'amazon', 'ebay', 'etsy', 'shopify', 'walmart', 'target',
                         'bestbuy', 'alibaba', 'wish'],
            'cosmetics': ['cosmetics', 'beauty', 'makeup', 'skincare', 'loreal', 'maybelline',
                         'sephora', 'ulta', 'revlon', 'covergirl', 'olay', 'clinique'],
            'automotive': ['auto', 'car', 'vehicle', 'motor', 'ford', 'toyota', 'honda',
                          'bmw', 'mercedes', 'audi', 'volkswagen', 'tesla', 'uber', 'lyft'],
            'food': ['food', 'restaurant', 'cafe', 'pizza', 'burger', 'starbucks',
                    'mcdonalds', 'subway', 'dominos', 'kfc', 'cocacola', 'pepsi'],
            'fashion': ['fashion', 'clothing', 'apparel', 'style', 'nike', 'adidas',
                       'zara', 'hm', 'gap', 'uniqlo', 'forever21', 'guess'],
            'media': ['news', 'media', 'tv', 'radio', 'streaming', 'netflix', 'youtube',
                     'spotify', 'cnn', 'bbc', 'espn', 'disney', 'hulu'],
            'healthcare': ['health', 'medical', 'hospital', 'care', 'pharmacy', 'cvs',
                          'walgreens', 'pfizer', 'johnson', 'abbott', 'medtronic'],
            'education': ['education', 'school', 'university', 'learning', 'course',
                         'harvard', 'stanford', 'mit', 'coursera', 'udemy', 'khan']
        }
        
        # Common brand family patterns
        self.brand_patterns = {
            'google': ['google', 'youtube', 'gmail', 'chrome', 'android', 'drive'],
            'microsoft': ['microsoft', 'windows', 'office', 'azure', 'teams', 'outlook'],
            'amazon': ['amazon', 'aws', 'prime', 'alexa', 'kindle', 'twitch'],
            'apple': ['apple', 'iphone', 'ipad', 'mac', 'ios', 'safari'],
            'meta': ['facebook', 'instagram', 'whatsapp', 'messenger', 'meta'],
            'disney': ['disney', 'pixar', 'marvel', 'starwars', 'espn', 'hulu']
        }

    def extract_brand_family(self, domain):
        """Extract brand family from domain name"""
        domain_lower = domain.lower()
        
        # Check for exact brand family matches
        for family, brands in self.brand_patterns.items():
            for brand in brands:
                if brand in domain_lower:
                    return family
        
        # Extract base brand from domain (remove common extensions)
        brand_name = domain_lower.replace('www.', '').replace('.com', '').replace('.org', '')
        brand_name = brand_name.replace('.net', '').replace('.co', '').split('.')[0]
        
        return brand_name if brand_name else 'unknown'

    def classify_industry(self, domain):
        """Classify industry based on domain and keywords"""
        domain_lower = domain.lower()
        scores = defaultdict(float)
        
        # Keyword-based classification
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in domain_lower:
                    scores[industry] += 1.0
        
        # Return the highest scoring industry or 'general' if no match
        if scores:
            return max(scores, key=scores.get)
        return 'general'
