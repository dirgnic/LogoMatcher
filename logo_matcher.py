import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
from PIL import Image
import hashlib
import urllib.parse
from collections import defaultdict
import json
import os
import re
from typing import List, Dict, Tuple, Optional
import time
import logging
import io

class LogoMatcher:
    def __init__(self, config_file: str = 'config.json'):
        self.logos = {}
        self.load_config(config_file)
        self.setup_logging()
        
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.similarity_threshold = config.get('similarity_threshold', 0.85)
            self.request_timeout = config.get('request_timeout', 15)
            self.request_delay = config.get('request_delay', 2)
            self.max_retries = config.get('max_retries', 3)
            self.max_logo_candidates = config.get('max_logo_candidates', 5)
            self.max_image_size_mb = config.get('max_image_size_mb', 10)
            self.user_agent = config.get('user_agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
            self.logo_selectors = config.get('logo_selectors', [])
            self.skip_domains = set(config.get('skip_problematic_domains', []))
            
        except FileNotFoundError:
            # Use defaults if config file not found
            self.similarity_threshold = 0.85
            self.request_timeout = 15
            self.request_delay = 2
            self.max_retries = 3
            self.max_logo_candidates = 5
            self.max_image_size_mb = 10
            self.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            self.logo_selectors = []
            self.skip_domains = set()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def clean_url(self, url: str) -> str:
        """Clean and validate URL"""
        if not url or not isinstance(url, str):
            self.logger.debug(f"Invalid URL type or empty: {url}")
            return ""
        
        # Remove whitespace and common issues
        url = url.strip()
        
        if not url:
            self.logger.debug("Empty URL after stripping")
            return ""
        
        # Remove protocol if present for cleaning
        clean_url = url
        if url.startswith(('http://', 'https://')):
            clean_url = url.split('://', 1)[1]
        
        # Remove www. prefix for consistency
        if clean_url.startswith('www.'):
            clean_url = clean_url[4:]
            
        # Remove trailing slashes and paths for domain validation
        domain = clean_url.split('/')[0]
        
        # Basic domain validation
        if not domain or '.' not in domain or len(domain) < 4:
            self.logger.debug(f"Invalid domain format: '{domain}' from '{url}'")
            return ""
            
        # Check for invalid characters
        if any(char in domain for char in [' ', '\t', '\n', '\r']):
            self.logger.debug(f"Domain contains invalid characters: '{domain}'")
            return ""
        
        # Check for obviously invalid domains
        if domain.count('.') < 1 or domain.startswith('.') or domain.endswith('.'):
            self.logger.debug(f"Malformed domain: '{domain}'")
            return ""
            
        return domain
        
    def extract_logo_urls(self, website_url: str) -> List[str]:
        """Extract potential logo URLs from a website"""
        try:
            # Clean and validate URL
            website_url = self.clean_url(website_url)
            if not website_url:
                return []
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Try HTTPS first, then HTTP if it fails
            for protocol in ['https://', 'http://']:
                try:
                    if not website_url.startswith(('http://', 'https://')):
                        full_url = protocol + website_url
                    else:
                        full_url = website_url
                    
                    response = requests.get(full_url, headers=headers, timeout=15, allow_redirects=True)
                    response.raise_for_status()
                    break
                except requests.exceptions.SSLError:
                    if protocol == 'https://':
                        continue  # Try HTTP
                    else:
                        raise
                except requests.exceptions.ConnectionError:
                    if protocol == 'https://':
                        continue  # Try HTTP
                    else:
                        raise
            
            soup = BeautifulSoup(response.content, 'html.parser')
            parsed_url = urllib.parse.urlparse(full_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            logo_urls = []
            
            # Common logo selectors
            logo_selectors = [
                'img[alt*="logo" i]',
                'img[src*="logo" i]',
                'img[class*="logo" i]',
                'img[id*="logo" i]',
                '.logo img',
                '#logo img',
                'header img',
                '.header img',
                '.navbar img',
                '.brand img',
                '.site-logo img',
                '.header-logo img',
                '.navbar-brand img'
            ]
            
            for selector in logo_selectors:
                elements = soup.select(selector)
                for img in elements:
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        # Convert relative URLs to absolute and validate
                        absolute_url = self.make_absolute_url(src, base_url, parsed_url.scheme)
                        if absolute_url and self.is_valid_image_url(absolute_url):
                            logo_urls.append(absolute_url)
            
            # Also check favicon
            favicon_links = soup.find_all('link', rel=lambda x: x and 'icon' in x.lower())
            for link in favicon_links:
                href = link.get('href')
                if href:
                    absolute_url = self.make_absolute_url(href, base_url, parsed_url.scheme)
                    if absolute_url and self.is_valid_image_url(absolute_url):
                        logo_urls.append(absolute_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_logos = []
            for url in logo_urls:
                if url not in seen:
                    seen.add(url)
                    unique_logos.append(url)
                    
            return unique_logos[:5]  # Limit to top 5 candidates
            
        except Exception as e:
            self.logger.error(f"Error extracting logos from {website_url}: {e}")
            return []
    
    def make_absolute_url(self, url: str, base_url: str, scheme: str) -> str:
        """Convert relative URL to absolute URL with better error handling"""
        if not url:
            return ""
            
        try:
            # Handle protocol-relative URLs
            if url.startswith('//'):
                return scheme + ':' + url
            
            # Handle absolute URLs
            if url.startswith(('http://', 'https://')):
                return url
            
            # Handle root-relative URLs
            if url.startswith('/'):
                return base_url + url
            
            # Handle relative URLs
            return base_url + '/' + url
            
        except Exception:
            return ""
    
    def is_valid_image_url(self, url: str) -> bool:
        """Check if URL looks like a valid image URL"""
        if not url:
            return False
            
        # Check for common image extensions
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.ico']
        url_lower = url.lower()
        
        # Check if URL has image extension
        has_image_ext = any(url_lower.endswith(ext) for ext in image_extensions)
        
        # Check if URL contains image-related keywords
        image_keywords = ['logo', 'brand', 'icon', 'favicon']
        has_image_keyword = any(keyword in url_lower for keyword in image_keywords)
        
        # Allow if it has image extension or image keyword
        return has_image_ext or has_image_keyword
    
    def download_image(self, image_url: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """Download and process image with retry logic"""
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive'
                }
                
                # Validate URL before attempting download
                if not self.is_valid_image_url(image_url):
                    self.logger.warning(f"Skipping invalid image URL: {image_url}")
                    return None
                
                response = requests.get(
                    image_url, 
                    headers=headers, 
                    timeout=self.request_timeout,
                    allow_redirects=True,
                    stream=True
                )
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image', 'octet-stream']):
                    self.logger.warning(f"URL does not serve image content: {image_url}")
                    return None
                
                # Check content length (avoid huge files)
                content_length = response.headers.get('content-length')
                max_size_bytes = self.max_image_size_mb * 1024 * 1024
                if content_length and int(content_length) > max_size_bytes:
                    self.logger.warning(f"Image too large: {image_url}")
                    return None
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(response.content))
                
                # Convert to RGB if necessary
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                elif img.mode == 'RGBA':
                    # Create white background for RGBA images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                
                # Convert to OpenCV format
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                return img_bgr
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout downloading image {image_url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error downloading image {image_url} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [404, 403, 401]:
                    self.logger.warning(f"HTTP error {e.response.status_code} for {image_url}")
                    break  # Don't retry for these errors
                else:
                    self.logger.warning(f"HTTP error downloading image {image_url} (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
            except Exception as e:
                self.logger.warning(f"Error downloading image {image_url} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        return None
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for comparison"""
        # Resize to standard size
        img_resized = cv2.resize(img, (128, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Normalize
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def compute_image_hash(self, img: np.ndarray) -> str:
        """Compute perceptual hash of image"""
        # Resize to 8x8
        small = cv2.resize(img, (8, 8))
        
        # Calculate mean
        mean = np.mean(small)
        
        # Create binary hash
        binary = small > mean
        
        # Convert to string
        hash_str = ''.join(['1' if pixel else '0' for pixel in binary.flatten()])
        
        return hash_str
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def compute_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute histogram similarity between images"""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare using correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return correlation
    
    def compute_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute structural similarity using template matching"""
        # Ensure same size
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # Template matching
        result = cv2.matchTemplate(img1_resized, img2_resized, cv2.TM_CCOEFF_NORMED)
        
        return result[0][0]
    
    def are_images_similar(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[bool, float]:
        """Determine if two images are similar using multiple metrics"""
        # Preprocess images
        proc1 = self.preprocess_image(img1)
        proc2 = self.preprocess_image(img2)
        
        # Compute perceptual hashes
        hash1 = self.compute_image_hash(proc1)
        hash2 = self.compute_image_hash(proc2)
        
        # Hamming distance (lower is more similar)
        hamming_dist = self.hamming_distance(hash1, hash2)
        hash_similarity = 1 - (hamming_dist / 64.0)  # 64 bits total
        
        # Histogram similarity
        hist_similarity = self.compute_histogram_similarity(proc1, proc2)
        
        # Structural similarity
        struct_similarity = self.compute_structural_similarity(proc1, proc2)
        
        # Combined similarity score
        combined_score = (hash_similarity * 0.4 + 
                         hist_similarity * 0.3 + 
                         struct_similarity * 0.3)
        
        is_similar = combined_score >= self.similarity_threshold
        
        return is_similar, combined_score
    
    def process_website(self, website_url: str) -> Dict:
        """Process a single website to extract and analyze its logo"""
        original_url = website_url
        self.logger.info(f"Processing website: {original_url}")
        
        result = {
            'website': original_url,
            'logo_found': False,
            'logo_url': None,
            'logo_data': None,
            'error': None,
            'attempts': 0,
            'skipped': False
        }
        
        try:
            # Clean URL first
            cleaned_url = self.clean_url(website_url)
            if not cleaned_url:
                result['error'] = 'Invalid URL format'
                return result
            
            # Check if domain should be skipped
            if cleaned_url in self.skip_domains:
                result['skipped'] = True
                result['error'] = 'Domain in skip list (known problematic)'
                self.logger.info(f"Skipping problematic domain: {cleaned_url}")
                return result
            
            # Extract logo URLs
            logo_urls = self.extract_logo_urls(cleaned_url)
            result['attempts'] = len(logo_urls)
            
            if not logo_urls:
                result['error'] = 'No logo URLs found'
                return result
            
            # Try each logo URL until we find a valid one
            for i, logo_url in enumerate(logo_urls):
                self.logger.debug(f"Trying logo URL {i+1}/{len(logo_urls)}: {logo_url}")
                img_data = self.download_image(logo_url, self.max_retries)
                if img_data is not None:
                    # Validate image has reasonable dimensions
                    if img_data.shape[0] > 10 and img_data.shape[1] > 10:
                        result['logo_found'] = True
                        result['logo_url'] = logo_url
                        result['logo_data'] = img_data
                        self.logger.info(f"Successfully extracted logo from {original_url}")
                        break
                    else:
                        self.logger.debug(f"Image too small: {img_data.shape}")
            
            if not result['logo_found']:
                result['error'] = 'Failed to download any valid logo images'
                
        except requests.exceptions.ConnectionError as e:
            result['error'] = f'Connection error: {str(e)[:100]}...'
        except requests.exceptions.Timeout as e:
            result['error'] = f'Timeout error: {str(e)[:100]}...'
        except Exception as e:
            result['error'] = f'Unexpected error: {str(e)[:100]}...'
            
        return result
    
    def group_similar_logos(self, processed_websites: List[Dict]) -> List[List[str]]:
        """Group websites with similar logos"""
        groups = []
        used_websites = set()
        
        valid_websites = [w for w in processed_websites if w['logo_found']]
        
        for i, website1 in enumerate(valid_websites):
            if website1['website'] in used_websites:
                continue
                
            current_group = [website1['website']]
            used_websites.add(website1['website'])
            
            for j, website2 in enumerate(valid_websites[i+1:], i+1):
                if website2['website'] in used_websites:
                    continue
                    
                is_similar, score = self.are_images_similar(
                    website1['logo_data'], 
                    website2['logo_data']
                )
                
                if is_similar:
                    current_group.append(website2['website'])
                    used_websites.add(website2['website'])
                    self.logger.info(f"Similar logos found: {website1['website']} <-> {website2['website']} (score: {score:.3f})")
            
            groups.append(current_group)
        
        return groups
    
    def run_analysis(self, websites: List[str]) -> Dict:
        """Run the complete logo matching analysis"""
        self.logger.info(f"Starting analysis of {len(websites)} websites")
        
        processed_websites = []
        error_counts = defaultdict(int)
        unparseable_domains = 0
        skipped_domains = 0
        
        for i, website in enumerate(websites, 1):
            self.logger.info(f"Progress: {i}/{len(websites)} - Processing {website}")
            
            # Check if domain is parseable before processing
            cleaned_url = self.clean_url(website)
            if not cleaned_url:
                unparseable_domains += 1
                result = {
                    'website': website,
                    'logo_found': False,
                    'logo_url': None,
                    'logo_data': None,
                    'error': 'Unparseable domain format',
                    'attempts': 0,
                    'skipped': False
                }
                processed_websites.append(result)
                error_counts['Unparseable domain format'] += 1
                continue
            
            result = self.process_website(website)
            processed_websites.append(result)
            
            # Track different types of issues
            if result.get('skipped', False):
                skipped_domains += 1
            elif not result['logo_found'] and result['error']:
                error_type = result['error'].split(':')[0]
                error_counts[error_type] += 1
            
            # Be respectful to servers with delay
            if i < len(websites):  # Don't delay after the last website
                time.sleep(self.request_delay)
        
        # Calculate statistics
        successful_extractions = sum(1 for w in processed_websites if w['logo_found'])
        extraction_rate = (successful_extractions / len(websites)) * 100
        
        # Group similar logos
        self.logger.info("Grouping similar logos...")
        groups = self.group_similar_logos(processed_websites)
        
        # Prepare results
        results = {
            'total_websites': len(websites),
            'successful_extractions': successful_extractions,
            'extraction_rate': extraction_rate,
            'unparseable_domains': unparseable_domains,
            'skipped_domains': skipped_domains,
            'failed_extractions': len(websites) - successful_extractions - skipped_domains,
            'groups': groups,
            'error_summary': dict(error_counts),
            'detailed_results': processed_websites
        }
        
        self.logger.info(f"Analysis complete. Extraction rate: {extraction_rate:.1f}%")
        self.logger.info(f"Unparseable domains: {unparseable_domains}")
        self.logger.info(f"Skipped domains: {skipped_domains}")
        self.logger.info(f"Found {len(groups)} groups")
        if error_counts:
            self.logger.info(f"Error summary: {dict(error_counts)}")
        
        return results
