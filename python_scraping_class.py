#!/usr/bin/env python3
"""
Python Logo Scraping & Visualization Engine
Handles all I/O operations: API calls, image processing, data management, and visualization
Delegates heavy Fourier mathematics to C++ backend for optimal performance
"""

import asyncio
import aiohttp
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import json
import time
import io
import hashlib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# C++ integration for Fourier mathematics
import sys
from pathlib import Path

# Add build directory to Python path
build_path = Path(__file__).parent / "build"
if build_path.exists() and str(build_path) not in sys.path:
    sys.path.insert(0, str(build_path))

try:
    import fourier_math_cpp  # This will be our C++ module
    CPP_AVAILABLE = True
    print("C++ Fourier module loaded - enhanced threading enabled")
except ImportError as e:
    print(f"C++ Fourier module not available: {e}")
    print("   Using Python fallback implementation")
    CPP_AVAILABLE = False
    # Fallback imports
    try:
        from scipy.fft import fft2, fftshift
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("scipy/sklearn not available - using basic implementations")
        fft2, fftshift, cosine_similarity = None, None, None

class LogoScrapingEngine:
    """
    High-performance logo scraping engine optimized for MacBook Pro 2024
    Handles API extraction, caching, and data management
    """
    
    def __init__(self, config_path: str = "logo_apis_config.json"):
        self.config = self._load_api_config(config_path)
        self.session = None
        self.cache_dir = Path("logo_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Performance settings for MacBook Pro 2024
        self.max_concurrent = 150  # Optimized for M3 Pro/Max
        self.batch_size = 500
        self.timeout_settings = {
            'total': 10,
            'connect': 3,
            'sock_read': 5
        }
    
    def load_parquet_data(self, parquet_file: str = "logos.snappy.parquet") -> List[str]:
        """Load website domains from parquet file"""
        try:
            df = pd.read_parquet(parquet_file, engine='pyarrow')
            domains = df['domain'].tolist()
            print(f"Loaded {len(domains)} domains from {parquet_file}")
            return domains
        except Exception as e:
            print(f"Error loading parquet file {parquet_file}: {e}")
            return []
    
    def load_existing_logo_data(self, pickle_file: str = "logo_extraction_results.pkl") -> list:
        """Load previously extracted logo data from pickle file - returns list of logo results"""
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"📋 Loading existing logo data from {pickle_file}")
            print(f"🔍 Data type: {type(data)}")
            
            # Handle different data structures like my_collab
            if isinstance(data, list):
                # Direct list of logo results 
                successful_logos = [r for r in data if r.get('logo_found', False)]
                print(f"✅ Loaded {len(successful_logos)}/{len(data)} successful logos from list")
                return data
                
            elif isinstance(data, dict):
                print(f"📊 Dict keys: {list(data.keys())}")
                # Try different possible keys
                for key in ['logo_results', 'successful_logos', 'results']:
                    if key in data:
                        results = data[key]
                        if isinstance(results, list):
                            successful_logos = [r for r in results if r.get('logo_found', False)]
                            print(f"✅ Loaded {len(successful_logos)}/{len(results)} successful logos from '{key}'")
                            return results
                        elif isinstance(results, dict):
                            # Convert dict to list
                            result_list = list(results.values())
                            successful_logos = [r for r in result_list if r.get('logo_found', False)]
                            print(f"✅ Loaded {len(successful_logos)}/{len(result_list)} successful logos from dict '{key}'")
                            return result_list
                
                # If no specific key found, return empty list
                print(f"⚠️ No recognized logo results found in dict")
                return []
            else:
                print(f"⚠️ Unexpected data type: {type(data)}")
                return []
                
        except Exception as e:
            print(f"❌ Error loading pickle file {pickle_file}: {e}")
            return []
    
    def _load_api_config(self, config_path: str) -> dict:
        """Load API configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using default APIs")
            return self._default_api_config()
    
    def _default_api_config(self) -> dict:
        """Fallback API configuration"""
        return {
            "logo_apis": [
                {
                    "name": "Clearbit",
                    "url": "https://logo.clearbit.com/{domain}",
                    "params": {},
                    "headers": {},
                    "timeout": 3,
                    "tier": 1
                },
                {
                    "name": "Google Favicon", 
                    "url": "https://www.google.com/s2/favicons",
                    "params": {"domain": "{domain}", "sz": "128"},
                    "headers": {},
                    "timeout": 2,
                    "tier": 1
                }
            ]
        }
    
    async def __aenter__(self):
        """Async context manager setup"""
        timeout = aiohttp.ClientTimeout(**self.timeout_settings)
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'LogoMatcher-Pro/2.0 (MacBook-Pro-2024)',
                'Accept': 'image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async session"""
        if self.session:
            await self.session.close()
    
    def _get_cache_path(self, domain: str) -> Path:
        """Generate cache file path for domain"""
        domain_hash = hashlib.sha256(domain.encode()).hexdigest()[:16]
        return self.cache_dir / f"{domain_hash}_{domain}.pkl"
    
    def _load_from_cache(self, domain: str) -> Optional[dict]:
        """Load logo data from cache if available"""
        cache_path = self._get_cache_path(domain)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                # Check if cache is less than 24 hours old
                if time.time() - cached_data['timestamp'] < 86400:
                    return cached_data
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, domain: str, logo_data: bytes, source: str):
        """Save logo data to cache"""
        try:
            cache_data = {
                'domain': domain,
                'logo_data': logo_data,
                'source': source,
                'timestamp': time.time()
            }
            cache_path = self._get_cache_path(domain)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Cache save failed for {domain}: {e}")
    
    def clean_domain(self, website: str) -> str:
        """Extract clean domain from URL"""
        if website.startswith(('http://', 'https://')):
            from urllib.parse import urlparse
            return urlparse(website).netloc
        return website.strip().lower()
    
    async def extract_single_logo(self, domain: str) -> Dict:
        """
        Extract logo for a single domain using API cascade
        Returns: {'domain': str, 'logo_data': bytes, 'source': str, 'success': bool}
        """
        clean_domain = self.clean_domain(domain)
        
        # Check cache first
        cached = self._load_from_cache(clean_domain)
        if cached:
            return {
                'domain': clean_domain,
                'logo_data': cached['logo_data'],
                'source': f"{cached['source']} (cached)",
                'success': True,
                'size': len(cached['logo_data'])
            }
        
        # Try each API in order of preference
        for api in self.config['logo_apis']:
            try:
                logo_data = await self._try_api_service(api, clean_domain)
                if logo_data and len(logo_data) > 1000:  # Minimum viable logo size
                    # Save to cache
                    self._save_to_cache(clean_domain, logo_data, api['name'])
                    
                    return {
                        'domain': clean_domain,
                        'logo_data': logo_data,
                        'source': api['name'],
                        'success': True,
                        'size': len(logo_data)
                    }
            except Exception as e:
                continue
        
        # No logo found
        return {
            'domain': clean_domain,
            'logo_data': None,
            'source': 'none',
            'success': False,
            'size': 0
        }
    
    async def _try_api_service(self, api_config: dict, domain: str) -> Optional[bytes]:
        """Try extracting logo from a single API service"""
        try:
            # Format URL and parameters
            url = api_config['url'].format(domain=domain)
            params = {}
            for key, value in api_config.get('params', {}).items():
                params[key] = str(value).format(domain=domain)
            
            # Make request with appropriate timeout
            timeout = aiohttp.ClientTimeout(total=api_config['timeout'])
            async with self.session.get(
                url,
                params=params,
                headers=api_config.get('headers', {}),
                timeout=timeout
            ) as response:
                
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        return await response.read()
        
        except Exception:
            pass
        
        return None
    
    async def extract_batch_logos(self, domains: List[str], progress_callback=None) -> List[Dict]:
        """
        Extract logos for multiple domains with progress tracking
        Optimized for MacBook Pro 2024 performance
        """
        print(f"Starting batch extraction for {len(domains)} domains")
        print(f"Using {self.max_concurrent} concurrent connections")
        
        results = []
        total_processed = 0
        start_time = time.time()
        
        # Process in batches to manage memory
        for i in range(0, len(domains), self.batch_size):
            batch = domains[i:i + self.batch_size]
            batch_start = time.time()
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def extract_with_semaphore(domain):
                async with semaphore:
                    return await self.extract_single_logo(domain)
            
            # Process batch concurrently
            batch_tasks = [extract_with_semaphore(domain) for domain in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions and add successful results
            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)
                    total_processed += 1
            
            # Progress reporting
            batch_time = time.time() - batch_start
            successful_in_batch = sum(1 for r in batch_results if isinstance(r, dict) and r.get('success'))
            
            print(f"Batch {i//self.batch_size + 1}: {successful_in_batch}/{len(batch)} successful "
                  f"({batch_time:.2f}s, {len(batch)/batch_time:.1f} domains/sec)")
            
            if progress_callback:
                progress_callback(total_processed, len(domains))
        
        # Final statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get('success'))
        
        print(f"Extraction complete: {successful}/{len(domains)} successful")
        print(f"Total time: {total_time:.2f}s ({len(domains)/total_time:.1f} domains/sec)")
        
        return results
    
    def preprocess_logo_for_analysis(self, logo_data: bytes) -> Optional[np.ndarray]:
        """
        Preprocess logo data for analysis
        Optimized preprocessing pipeline for C++ integration
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(logo_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to standard size (128x128 for optimal C++ processing)
            image = image.resize((128, 128), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale for Fourier analysis
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            # Normalize to [0, 1] range for C++ consistency
            return img_gray.astype(np.float32) / 255.0
            
        except Exception as e:
            print(f"Error preprocessing logo: {e}")
            return None
    
    def save_extraction_results(self, results: List[Dict], output_path: str):
        """Save extraction results to multiple formats"""
        # Create DataFrame
        df_data = []
        for result in results:
            df_data.append({
                'domain': result['domain'],
                'success': result['success'],
                'source': result['source'],
                'size_bytes': result.get('size', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        csv_path = output_path.replace('.pkl', '.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as pickle (with logo data)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {csv_path} and {output_path}")


class LogoVisualizationEngine:
    """
    Advanced visualization engine for logo analysis results
    Creates publication-quality charts and reports
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib for high-quality output
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def create_extraction_performance_chart(self, results: List[Dict]) -> str:
        """Create performance analysis chart for logo extraction"""
        
        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.get('success'))
        
        # Source breakdown
        source_counts = defaultdict(int)
        for result in results:
            if result.get('success'):
                source = result['source'].replace(' (cached)', '')
                source_counts[source] += 1
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Success Rate Pie Chart
        ax1.pie([successful, total - successful], 
                labels=[f'Success ({successful})', f'Failed ({total - successful})'],
                colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Logo Extraction Success Rate', fontsize=14, fontweight='bold')
        
        # 2. Source Distribution
        sources = list(source_counts.keys())
        counts = list(source_counts.values())
        bars = ax2.bar(sources, counts, color=plt.cm.Set3(np.linspace(0, 1, len(sources))))
        ax2.set_title('Successful Extractions by Source', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Logos')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 3. Logo Size Distribution
        sizes = [r.get('size', 0) for r in results if r.get('success') and r.get('size', 0) > 0]
        if sizes:
            ax3.hist(sizes, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
            ax3.set_title('Logo File Size Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Size (bytes)')
            ax3.set_ylabel('Frequency')
            ax3.axvline(np.mean(sizes), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(sizes):.0f} bytes')
            ax3.legend()
        
        # 4. Performance Metrics Summary
        ax4.axis('off')
        metrics_text = f"""
        EXTRACTION PERFORMANCE SUMMARY
        
        Total Domains Processed: {total:,}
        Successful Extractions: {successful:,}
        Success Rate: {(successful/total)*100:.1f}%
        
        SOURCE PERFORMANCE:
        """
        
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            success_rate = (count / successful) * 100
            metrics_text += f"  • {source}: {count} ({success_rate:.1f}%)\n"
        
        if sizes:
            metrics_text += f"""
        LOGO STATISTICS:
        Average Size: {np.mean(sizes):.0f} bytes
        Size Range: {min(sizes):,} - {max(sizes):,} bytes
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_dir / "extraction_performance_analysis.png"
        plt.savefig(chart_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Performance chart saved to {chart_path}")
        return str(chart_path)
    
    def create_similarity_heatmap(self, similarity_matrix: np.ndarray, domain_names: List[str]) -> str:
        """Create similarity heatmap visualization"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(similarity_matrix, 
                   xticklabels=domain_names[:50],  # Limit for readability
                   yticklabels=domain_names[:50],
                   cmap='viridis', 
                   center=0.5,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'Similarity Score'})
        
        ax.set_title('Logo Similarity Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        chart_path = self.output_dir / "similarity_heatmap.png"
        plt.savefig(chart_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Similarity heatmap saved to {chart_path}")
        return str(chart_path)
    
    def create_cluster_visualization(self, clusters: List[List[str]], similarity_scores: List[float]) -> str:
        """Create cluster analysis visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Cluster Size Distribution
        cluster_sizes = [len(cluster) for cluster in clusters]
        ax1.hist(cluster_sizes, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cluster Size (number of logos)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(cluster_sizes), color='red', linestyle='--',
                   label=f'Mean: {np.mean(cluster_sizes):.1f}')
        ax1.legend()
        
        # 2. Similarity Score Distribution
        ax2.hist(similarity_scores, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_title('Similarity Score Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Similarity Score')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(similarity_scores), color='green', linestyle='--',
                   label=f'Mean: {np.mean(similarity_scores):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
        chart_path = self.output_dir / "cluster_analysis.png"
        plt.savefig(chart_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Cluster analysis saved to {chart_path}")
        return str(chart_path)


class LogoAnalysisPipeline:
    """
    Main pipeline orchestrator that combines scraping, C++ analysis, and visualization
    """
    
    def __init__(self, config_path: str = "logo_apis_config.json"):
        self.scraper = LogoScrapingEngine(config_path)
        self.visualizer = LogoVisualizationEngine()
        self.cpp_available = CPP_AVAILABLE
    
    async def run_analysis_from_parquet(self, 
                                      parquet_file: str = "logos.snappy.parquet",
                                      pickle_file: str = "logo_extraction_results.pkl",
                                      create_visualizations: bool = True,
                                      similarity_threshold: float = 0.45) -> Dict:
        """
        Run analysis using existing parquet domains and pickle logo data
        
        Args:
            parquet_file: Parquet file with website domains
            pickle_file: Pickle file with extracted logo data
            create_visualizations: Whether to generate charts
            similarity_threshold: Threshold for similarity clustering
            
        Returns:
            Complete analysis results dictionary
        """
        
        print("LOGO ANALYSIS PIPELINE - USING EXISTING DATA")
        print("=" * 60)
        
        total_start = time.time()
        
        # Load existing data
        print("\n1. LOADING EXISTING DATA")
        domains = self.scraper.load_parquet_data(parquet_file)
        logo_results_list = self.scraper.load_existing_logo_data(pickle_file)
        
        if not domains or not logo_results_list:
            print("ERROR: Could not load required data files")
            return {}
        
        # Filter successful logos (list format like my_collab)
        successful_logos = [r for r in logo_results_list if r.get('logo_found', False)]
        success_rate = len(successful_logos) / len(logo_results_list) * 100 if logo_results_list else 0
        
        print(f"✅ Loaded {len(successful_logos)}/{len(logo_results_list)} successful logos ({success_rate:.1f}% success)")
        
        if len(successful_logos) < 2:
            print("ERROR: Need at least 2 successful logos for analysis")
            return {}
        
        print(f"Processing {len(successful_logos)} successful logo extractions...")
        
        # Phase 2: Fourier Analysis (C++ Backend)
        if self.cpp_available and successful_logos:
            print("\n2. FOURIER ANALYSIS PHASE (C++ Backend)")
            
            try:
                import fourier_math_cpp
                
                # Prepare logo images for C++ analysis 
                logo_images = []
                domain_mapping = {}
                processed_logos = 0
                
                print("Preparing logo images for C++ analysis...")
                
                for logo_data in successful_logos:
                    if isinstance(logo_data, dict):
                        domain = logo_data.get('domain', logo_data.get('website', ''))
                        logo_bytes = logo_data.get('logo_data')
                        
                        if logo_bytes and domain and len(logo_bytes) > 100:  # Basic size check
                            try:
                                # Add valid logo image (skip strict validation for now)
                                logo_images.append(logo_bytes)
                                domain_mapping[processed_logos] = domain
                                processed_logos += 1
                                
                                if processed_logos % 500 == 0:
                                    print(f"  Processed {processed_logos}/{len(successful_logos)} logos...")
                                    
                            except Exception as e:
                                continue  # Skip failed images
                
                print(f"✅ Prepared {len(logo_images)}/{len(successful_logos)} valid logo images")
                
                if len(logo_images) < 2:
                    print("ERROR: Need at least 2 valid logo images for similarity analysis")
                    return {}
                
                print(f"Computing similarity matrix for {len(logo_images)} logos...")
                
                # Decode PNG bytes to image arrays for C++ processing
                processed_images = []
                for i, logo_bytes in enumerate(logo_images):
                    try:
                        # Convert bytes to PIL Image, convert to grayscale for C++ processing
                        image = Image.open(io.BytesIO(logo_bytes)).convert('L')  # L = grayscale
                        img_array = np.array(image, dtype=np.float64)  # C++ expects double
                        processed_images.append(img_array)
                    except Exception as e:
                        print(f"Failed to decode image {i}: {e}")
                        continue
                
                print(f"Successfully processed {len(processed_images)} images for C++ analysis")
                print(f"Domain mapping has {len(domain_mapping)} entries")
                print(f"Processed images count: {len(processed_images)}")
                
                start_time = time.time()
                similarity_results = fourier_math_cpp.compute_similarity_matrix(processed_images, 0.45)
                cpp_time = time.time() - start_time
                
                print(f"C++ similarity computation completed in {cpp_time:.2f}s")
                print(f"Performance: {len(logo_images)/cpp_time:.1f} comparisons/second")
                
                # Phase 3: Clustering & Analysis
                print("\n3. CLUSTERING & SIMILARITY ANALYSIS")
                
                # Get domain list in the same order as processed images
                # Note: C++ module might filter some images, so we need to match the matrix size
                expected_size = similarity_results.shape[0]
                domain_list = [domain_mapping[i] for i in range(expected_size)]
                
                if len(domain_list) != expected_size:
                    print(f"WARNING: Adjusting domain list from {len(processed_images)} to {expected_size} to match matrix")
                
                clusters, similar_pairs = self._analyze_similarity_matrix(
                    similarity_results, domain_list, similarity_threshold
                )
                
                # Phase 4: Visualization
                if create_visualizations:
                    print("\n4. VISUALIZATION GENERATION")
                    await self._create_analysis_visualizations(
                        successful_logos, clusters, similar_pairs, cpp_time
                    )
                
                total_time = time.time() - total_start
                
                # Generate final report
                results = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_domains': len(domains),
                    'analyzed_logos': len(successful_logos),
                    'valid_logos': len(logo_images),
                    'threshold_used': similarity_threshold,
                    'similar_pairs': similar_pairs,
                    'clusters': clusters,
                    'total_clustered_websites': sum(len(cluster) for cluster in clusters),
                    'performance_metrics': {
                        'cpp_computation_time': cpp_time,
                        'total_pipeline_time': total_time,
                        'logos_per_second': len(logo_images) / cpp_time if cpp_time > 0 else 0
                    }
                }
                
                print(f"\nANALYSIS COMPLETE - Total Time: {total_time:.1f}s")
                print(f"Processed: {len(successful_logos)} logos")
                print(f"Found: {len(similar_pairs)} similar pairs")
                print(f"Clusters: {len(clusters)} groups")
                print(f"Clustered websites: {results['total_clustered_websites']}")
                
                return results
                
            except ImportError as e:
                print(f"C++ module not available: {e}")
                print("Falling back to Python-only analysis...")
                return await self._python_only_analysis(successful_logos, similarity_threshold)
        
        else:
            print("\n2. PYTHON-ONLY ANALYSIS (C++ not available)")
            return await self._python_only_analysis(successful_logos, similarity_threshold)
    
    async def run_complete_analysis(self, 
                                  websites: List[str], 
                                  create_visualizations: bool = True,
                                  similarity_threshold: float = 0.45) -> Dict:
        """
        Run complete logo analysis pipeline
        
        Args:
            websites: List of website domains/URLs
            create_visualizations: Whether to generate charts
            similarity_threshold: Threshold for similarity clustering
            
        Returns:
            Complete analysis results dictionary
        """
        
        print("LOGO ANALYSIS PIPELINE - PYTHON + C++ HYBRID")
        print("=" * 60)
        
        total_start = time.time()
        
        # Phase 1: Logo Extraction (Python)
        print("\n1. LOGO EXTRACTION PHASE (Python)")
        print("-" * 40)
        
        async with self.scraper:
            extraction_results = await self.scraper.extract_batch_logos(websites)
        
        successful_logos = [r for r in extraction_results if r.get('success')]
        
        if len(successful_logos) < 2:
            print("Need at least 2 logos for analysis")
            return {'status': 'failed', 'reason': 'insufficient_logos'}
        
        print(f"Successfully extracted {len(successful_logos)} logos")
        
        # Phase 2: Feature Extraction & Similarity Analysis
        print("\n2. FOURIER ANALYSIS PHASE")
        print("-" * 40)
        
        if self.cpp_available:
            print("Using C++ backend for Fourier mathematics")
            similarity_results = self._cpp_similarity_analysis(successful_logos, similarity_threshold)
        else:
            print("Using Python fallback for Fourier mathematics")
            similarity_results = self._python_similarity_analysis(successful_logos, similarity_threshold)
        
        # Phase 3: Visualization (Python)
        if create_visualizations:
            print("\n3. VISUALIZATION PHASE (Python)")
            print("-" * 40)
            
            # Create performance chart
            perf_chart = self.visualizer.create_extraction_performance_chart(extraction_results)
            
            # Create similarity visualizations if we have enough data
            if len(similarity_results['clusters']) > 0:
                cluster_chart = self.visualizer.create_cluster_visualization(
                    similarity_results['clusters'],
                    similarity_results['similarity_scores']
                )
        
        # Final Results
        total_time = time.time() - total_start
        
        print(f"\nANALYSIS COMPLETE")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Found {len(similarity_results['clusters'])} logo clusters")
        
        return {
            'status': 'success',
            'total_time': total_time,
            'extraction_results': extraction_results,
            'similarity_results': similarity_results,
            'performance_metrics': {
                'total_websites': len(websites),
                'successful_extractions': len(successful_logos),
                'success_rate': len(successful_logos) / len(websites),
                'clusters_found': len(similarity_results['clusters'])
            }
        }
    
    def _analyze_similarity_matrix(self, similarity_matrix: np.ndarray, 
                                  domain_list: list, threshold: float) -> tuple:
        """Analyze similarity matrix from C++ module and create clusters with size constraints"""
        similar_pairs = []
        clusters = []
        
        n = len(domain_list)
        if similarity_matrix.shape != (n, n):
            print(f"WARNING: Matrix shape {similarity_matrix.shape} doesn't match domain count {n}")
            return [], []
        
        # Extract similar pairs above threshold
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    similar_pairs.append({
                        'domain1': domain_list[i],
                        'domain2': domain_list[j],
                        'similarity': float(similarity)
                    })
        
        print(f"Found {len(similar_pairs)} similar pairs above threshold {threshold}")
        
        # Natural clustering based on logo similarity (no forced target)
        clusters = self._create_natural_similarity_clusters(similar_pairs, max_cluster_size=50)
        
        print(f"Created {len(clusters)} natural similarity clusters")
        
        return clusters, similar_pairs
    
    def _create_natural_similarity_clusters(self, similar_pairs: list, max_cluster_size: int = 50) -> list:
        """Create natural clusters based purely on logo similarity without forcing a target count"""
        
        # Build similarity graph
        from collections import defaultdict, deque
        
        similarity_graph = defaultdict(list)
        all_domains = set()
        
        for pair in similar_pairs:
            d1, d2 = pair['domain1'], pair['domain2']
            similarity = pair['similarity']
            similarity_graph[d1].append((d2, similarity))
            similarity_graph[d2].append((d1, similarity))
            all_domains.add(d1)
            all_domains.add(d2)
        
        print(f"Building natural clusters from {len(all_domains)} domains with similarities...")
        
        # Find connected components using BFS (natural clusters)
        visited = set()
        natural_clusters = []
        
        for domain in all_domains:
            if domain not in visited:
                # BFS to find connected component
                component = []
                queue = deque([domain])
                visited.add(domain)
                
                while queue:
                    current = queue.popleft()
                    component.append(current)
                    
                    for neighbor, _ in similarity_graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                # Only keep clusters with multiple domains (meaningful groups)
                if len(component) >= 2:
                    natural_clusters.append(component)
        
        print(f"Found {len(natural_clusters)} natural connected components")
        
        # Only apply size constraints (split overly large clusters)
        final_clusters = []
        
        # Sort clusters by size for analysis
        natural_clusters.sort(key=len, reverse=True)
        cluster_sizes = [len(c) for c in natural_clusters]
        print(f"Natural cluster sizes: {cluster_sizes[:15]}{'...' if len(cluster_sizes) > 15 else ''}")
        
        for cluster in natural_clusters:
            if len(cluster) <= max_cluster_size:
                # Perfect size, keep as natural cluster
                final_clusters.append(cluster)
            else:
                # Split overly large cluster but preserve natural groupings
                print(f"Splitting large cluster of size {len(cluster)} (max: {max_cluster_size})")
                split_clusters = self._split_large_cluster(cluster, similarity_graph, max_cluster_size)
                final_clusters.extend(split_clusters)
        
        # Final statistics
        final_sizes = sorted([len(c) for c in final_clusters], reverse=True)
        print(f"Final natural clusters: {len(final_clusters)}")
        print(f"Cluster size distribution: {final_sizes[:15]}{'...' if len(final_sizes) > 15 else ''}")
        print(f"Average cluster size: {sum(final_sizes)/len(final_sizes):.1f}")
        print(f"Largest cluster: {max(final_sizes)} domains")
        print(f"Smallest cluster: {min(final_sizes)} domains")
        
        return final_clusters

    def _create_size_constrained_clusters(self, similar_pairs: list, target_clusters: int = 37, max_cluster_size: int = 50) -> list:
        """Create clusters with size constraints to achieve target number"""
        
        # Build similarity graph
        from collections import defaultdict, deque
        
        similarity_graph = defaultdict(list)
        all_domains = set()
        
        for pair in similar_pairs:
            d1, d2 = pair['domain1'], pair['domain2']
            similarity = pair['similarity']
            similarity_graph[d1].append((d2, similarity))
            similarity_graph[d2].append((d1, similarity))
            all_domains.add(d1)
            all_domains.add(d2)
        
        # Start with connected components using BFS
        visited = set()
        initial_clusters = []
        
        for domain in all_domains:
            if domain not in visited:
                # BFS to find connected component
                component = []
                queue = deque([domain])
                visited.add(domain)
                
                while queue:
                    current = queue.popleft()
                    component.append(current)
                    
                    for neighbor, _ in similarity_graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                if len(component) >= 2:
                    initial_clusters.append(component)
        
        print(f"Initial clusters: {len(initial_clusters)} (before size constraints)")
        
        # Apply size constraints to get target number of clusters
        final_clusters = []
        
        # Sort clusters by size (largest first)
        initial_clusters.sort(key=len, reverse=True)
        
        cluster_sizes = [len(c) for c in initial_clusters]
        print(f"Initial cluster sizes: {cluster_sizes[:10]}{'...' if len(cluster_sizes) > 10 else ''}")
        
        for cluster in initial_clusters:
            if len(cluster) <= max_cluster_size:
                # Small enough cluster, keep as is
                final_clusters.append(cluster)
            else:
                # Split large cluster into smaller ones
                split_clusters = self._split_large_cluster(cluster, similarity_graph, max_cluster_size)
                final_clusters.extend(split_clusters)
        
        # If we have too many clusters, merge smallest ones
        while len(final_clusters) > target_clusters and len(final_clusters) > 1:
            final_clusters.sort(key=len)
            # Merge two smallest clusters
            smallest = final_clusters.pop(0)
            second_smallest = final_clusters.pop(0)
            merged = smallest + second_smallest
            final_clusters.append(merged)
        
        final_cluster_sizes = sorted([len(c) for c in final_clusters], reverse=True)
        print(f"Final cluster sizes: {final_cluster_sizes[:10]}{'...' if len(final_cluster_sizes) > 10 else ''}")
        
        return final_clusters
    
    def _split_large_cluster(self, large_cluster: list, similarity_graph: dict, max_size: int) -> list:
        """Split a large cluster into smaller subclusters based on highest similarities"""
        
        if len(large_cluster) <= max_size:
            return [large_cluster]
        
        # Use greedy approach: start with highest similarity pairs
        pairs_in_cluster = []
        
        for i, domain1 in enumerate(large_cluster):
            for domain2, similarity in similarity_graph[domain1]:
                if domain2 in large_cluster and large_cluster.index(domain2) > i:
                    pairs_in_cluster.append((domain1, domain2, similarity))
        
        # Sort by similarity (highest first)
        pairs_in_cluster.sort(key=lambda x: x[2], reverse=True)
        
        # Greedily form subclusters
        subclusters = []
        used_domains = set()
        
        for domain1, domain2, similarity in pairs_in_cluster:
            # Find existing subclusters for these domains
            cluster1_idx = None
            cluster2_idx = None
            
            for i, subcluster in enumerate(subclusters):
                if domain1 in subcluster:
                    cluster1_idx = i
                if domain2 in subcluster:
                    cluster2_idx = i
            
            if cluster1_idx is None and cluster2_idx is None:
                # Create new subcluster
                if len(subclusters) * max_size < len(large_cluster):  # Still room for new clusters
                    subclusters.append([domain1, domain2])
                    used_domains.add(domain1)
                    used_domains.add(domain2)
            elif cluster1_idx is not None and cluster2_idx is None:
                # Add domain2 to existing cluster if room
                if len(subclusters[cluster1_idx]) < max_size:
                    subclusters[cluster1_idx].append(domain2)
                    used_domains.add(domain2)
            elif cluster1_idx is None and cluster2_idx is not None:
                # Add domain1 to existing cluster if room
                if len(subclusters[cluster2_idx]) < max_size:
                    subclusters[cluster2_idx].append(domain1)
                    used_domains.add(domain1)
            elif cluster1_idx != cluster2_idx:
                # Merge clusters if total size <= max_size
                total_size = len(subclusters[cluster1_idx]) + len(subclusters[cluster2_idx])
                if total_size <= max_size:
                    subclusters[cluster1_idx].extend(subclusters[cluster2_idx])
                    subclusters.pop(cluster2_idx)
        
        # Add remaining domains to smallest subclusters
        remaining_domains = [d for d in large_cluster if d not in used_domains]
        
        for domain in remaining_domains:
            if subclusters:
                # Add to smallest subcluster with room
                subclusters.sort(key=len)
                for subcluster in subclusters:
                    if len(subcluster) < max_size:
                        subcluster.append(domain)
                        break
                else:
                    # No room in existing subclusters, create new one
                    subclusters.append([domain])
            else:
                # No subclusters yet, create first one
                subclusters.append([domain])
        
        # Filter out single-domain clusters
        return [sc for sc in subclusters if len(sc) >= 2]
    
    def _analyze_similarity_results(self, similarity_results: dict, threshold: float) -> tuple:
        """Legacy method - kept for backward compatibility"""
        similar_pairs = []
        clusters = []
        
        # Extract similar pairs above threshold
        for (domain1, domain2), similarity in similarity_results.items():
            if similarity >= threshold:
                similar_pairs.append({
                    'domain1': domain1,
                    'domain2': domain2,
                    'similarity': float(similarity)
                })
        
        # Simple clustering: group connected domains
        domain_groups = {}
        group_id = 0
        
        for pair in similar_pairs:
            d1, d2 = pair['domain1'], pair['domain2']
            
            # Find existing groups for these domains
            g1 = domain_groups.get(d1)
            g2 = domain_groups.get(d2)
            
            if g1 is None and g2 is None:
                # New cluster
                domain_groups[d1] = group_id
                domain_groups[d2] = group_id
                group_id += 1
            elif g1 is not None and g2 is None:
                # Add d2 to d1's group
                domain_groups[d2] = g1
            elif g1 is None and g2 is not None:
                # Add d1 to d2's group
                domain_groups[d1] = g2
            elif g1 != g2:
                # Merge groups
                old_group = g2
                new_group = g1
                for domain, gid in domain_groups.items():
                    if gid == old_group:
                        domain_groups[domain] = new_group
        
        # Convert to cluster list
        group_to_domains = {}
        for domain, gid in domain_groups.items():
            if gid not in group_to_domains:
                group_to_domains[gid] = []
            group_to_domains[gid].append(domain)
        
        clusters = [domains for domains in group_to_domains.values() if len(domains) >= 2]
        
        return clusters, similar_pairs
    
    async def _create_analysis_visualizations(self, logos: dict, clusters: list, 
                                            similar_pairs: list, cpp_time: float):
        """Create visualizations for analysis results"""
        try:
            # Create performance chart
            await self.visualizer.create_performance_chart({
                'cpp_computation_time': cpp_time,
                'total_logos': len(logos),
                'processing_rate': len(logos) / cpp_time if cpp_time > 0 else 0
            })
            
            # Create cluster analysis chart
            if clusters:
                await self.visualizer.create_cluster_analysis(clusters)
            
            # Create similarity distribution chart
            if similar_pairs:
                similarities = [pair['similarity'] for pair in similar_pairs]
                await self.visualizer.create_similarity_distribution(similarities)
                
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    async def _python_only_analysis(self, logos: dict, threshold: float) -> dict:
        """Fallback Python-only similarity analysis"""
        print("Running basic Python similarity analysis...")
        
        # Simple feature comparison using existing Fourier features
        similar_pairs = []
        features_list = []
        domains_list = []
        
        for domain, logo_data in logos.items():
            if isinstance(logo_data, dict) and 'fourier_features' in logo_data:
                features = logo_data['fourier_features']
                if isinstance(features, np.ndarray) and features.size > 0:
                    features_list.append(features.flatten())
                    domains_list.append(domain)
        
        print(f"Comparing {len(features_list)} feature vectors...")
        
        # Pairwise comparison
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                similarity = np.corrcoef(features_list[i], features_list[j])[0, 1]
                if not np.isnan(similarity) and similarity >= threshold:
                    similar_pairs.append({
                        'domain1': domains_list[i],
                        'domain2': domains_list[j],
                        'similarity': float(similarity)
                    })
        
        clusters, _ = self._analyze_similarity_results(
            {(p['domain1'], p['domain2']): p['similarity'] for p in similar_pairs}, 
            threshold
        )
        
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analyzed_logos': len(logos),
            'valid_logos': len(features_list),
            'similar_pairs': similar_pairs,
            'clusters': clusters,
            'total_clustered_websites': sum(len(cluster) for cluster in clusters),
            'method': 'python_correlation'
        }
    
    def _cpp_similarity_analysis(self, successful_logos: List[Dict], threshold: float) -> Dict:
        """Use C++ backend for Fourier analysis and similarity computation"""
        
        # Preprocess logos for C++ analysis
        processed_images = []
        valid_logos = []
        
        for logo_data in successful_logos:
            img = self.scraper.preprocess_logo_for_analysis(logo_data['logo_data'])
            if img is not None:
                processed_images.append(img)
                valid_logos.append(logo_data)
        
        if len(processed_images) < 2:
            return {'clusters': [], 'similarity_scores': [], 'valid_logos': []}
        
        # Use C++ module for heavy Fourier mathematics
        try:
            # This would call our C++ module
            similarity_matrix = fourier_math_cpp.compute_similarity_matrix(processed_images)
            clusters = fourier_math_cpp.find_clusters(similarity_matrix, threshold)
            
            # Extract similarity scores for visualization
            similarity_scores = []
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix[i])):
                    if similarity_matrix[i][j] >= threshold:
                        similarity_scores.append(similarity_matrix[i][j])
            
            return {
                'clusters': clusters,
                'similarity_scores': similarity_scores,
                'similarity_matrix': similarity_matrix,
                'valid_logos': valid_logos
            }
            
        except Exception as e:
            print(f"C++ analysis failed: {e}, falling back to Python")
            return self._python_similarity_analysis(successful_logos, threshold)
    
    def _python_similarity_analysis(self, successful_logos: List[Dict], threshold: float) -> Dict:
        """Python fallback for Fourier analysis"""
        
        print("Running Python-based Fourier analysis...")
        
        # Simple similarity analysis using basic image hashing
        valid_logos = []
        hashes = []
        
        for logo_data in successful_logos:
            img = self.scraper.preprocess_logo_for_analysis(logo_data['logo_data'])
            if img is not None:
                # Simple perceptual hash
                img_small = cv2.resize(img, (8, 8))
                hash_val = img_small.flatten()
                hashes.append(hash_val)
                valid_logos.append(logo_data)
        
        # Compute pairwise similarities
        similarity_scores = []
        clusters = []
        
        if len(hashes) >= 2:
            # Simple correlation-based similarity
            for i in range(len(hashes)):
                for j in range(i+1, len(hashes)):
                    similarity = np.corrcoef(hashes[i], hashes[j])[0, 1]
                    if not np.isnan(similarity) and similarity >= threshold:
                        similarity_scores.append(similarity)
                        # Simple clustering: pair similar logos
                        clusters.append([valid_logos[i]['domain'], valid_logos[j]['domain']])
        
        return {
            'clusters': clusters,
            'similarity_scores': similarity_scores,
            'valid_logos': valid_logos
        }


# Example usage function
async def main_demo():
    """Demo function showing how to use the new architecture"""
    
    # Sample websites
    test_websites = [
        'google.com',
        'microsoft.com', 
        'apple.com',
        'amazon.com',
        'facebook.com'
    ]
    
    # Initialize pipeline
    pipeline = LogoAnalysisPipeline()
    
    # Run analysis
    results = await pipeline.run_complete_analysis(
        websites=test_websites,
        create_visualizations=True,
        similarity_threshold=0.45
    )
    
    print(f"\nAnalysis Results:")
    print(f"Status: {results['status']}")
    print(f"Success Rate: {results['performance_metrics']['success_rate']:.1%}")
    print(f"Clusters Found: {results['performance_metrics']['clusters_found']}")


if __name__ == "__main__":
    asyncio.run(main_demo())
