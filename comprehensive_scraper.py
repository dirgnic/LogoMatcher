#!/usr/bin/env python3
"""
Comprehensive Logo Scraping Pipeline
Uses the enhanced API extraction from my_collab.ipynb to achieve >98% success rate
"""

import asyncio
import aiohttp
import numpy as np
from PIL import Image
import pandas as pd
import json
import time
import io
import os
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import pickle
import concurrent.futures
import threading
from queue import Queue
import random

# Enhanced API Logo Extractor (from my_collab.ipynb)
class EnhancedAPILogoExtractor:
    """Enhanced multi-tier logo extraction with comprehensive API coverage and disk caching"""
    
    def __init__(self, use_cache=True, cache_dir="logo_cache"):
        self.session = None
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        if self.use_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Load API configs from JSON file + built-in configs
        self.api_configs = self._load_api_configs()
    
    def _load_api_configs(self):
        """Load API configurations from logo_apis_config.json and built-in configs"""
        configs = {}
        
        # Built-in high-quality configs (tested and working)
        built_in_configs = {
            'clearbit': {
                'url_template': 'https://logo.clearbit.com/{domain}',
                'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                'tier': 1
            },
            'logo_dev': {
                'url_template': 'https://img.logo.dev/{domain}?token=pk_X1XmKr69T5ufyVVfz5lP6Q',
                'headers': {'User-Agent': 'LogoFetcher/1.0'},
                'tier': 1
            },
            'google_favicon': {
                'url_template': 'https://www.google.com/s2/favicons?domain={domain}&sz=128',
                'headers': {},
                'tier': 2
            },
            'besticon': {
                'url_template': 'https://besticon-demo.herokuapp.com/icon?url={domain}&size=128',
                'headers': {},
                'tier': 3
            },
            'duckduckgo': {
                'url_template': 'https://icons.duckduckgo.com/ip3/{domain}.ico',
                'headers': {},
                'tier': 3
            },
            'iconhorse': {
                'url_template': 'https://icon.horse/icon/{domain}',
                'headers': {},
                'tier': 4
            },
            'favicon_io': {
                'url_template': 'https://favicon.io/favicon.ico?domain={domain}',
                'headers': {},
                'tier': 4
            }
        }
        
        configs.update(built_in_configs)
        
        # Try to load additional configs from JSON file
        try:
            with open('logo_apis_config.json', 'r') as f:
                json_config = json.load(f)
                
            for api in json_config.get('logo_apis', []):
                name = api.get('name', '').lower().replace(' ', '_')
                if name and name not in configs:  # Don't override built-in configs
                    
                    # Build URL template
                    url = api.get('url', '')
                    params = api.get('params', {})
                    if params:
                        # Convert params to query string
                        param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
                        url += '?' + param_str
                    
                    configs[name] = {
                        'url_template': url,
                        'headers': api.get('headers', {}),
                        'tier': api.get('tier', 5),
                        'timeout': api.get('timeout', 5)
                    }
                    
            print(f"Loaded {len(configs)} API configurations ({len(built_in_configs)} built-in + {len(configs) - len(built_in_configs)} from config file)")
            
        except Exception as e:
            print(f"Could not load logo_apis_config.json: {e}. Using built-in configs only.")
        
        return configs
    
    def _get_cache_path(self, domain):
        """Get cache file path for a domain"""
        if not self.use_cache:
            return None
        # Create a safe filename using domain hash
        domain_hash = hashlib.md5(domain.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{domain_hash}_{domain}.pkl")
    
    def _load_from_cache(self, domain):
        """Load logo from cache if available"""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(domain)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_result = pickle.load(f)
                    # Verify it's a successful result
                    if cached_result.get('success') and cached_result.get('logo_data'):
                        return cached_result
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, domain, result):
        """Save successful logo extraction to cache"""
        if not self.use_cache or not result.get('success'):
            return
            
        cache_path = self._get_cache_path(domain)
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            except Exception:
                pass
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def extract_domain(self, website: str) -> str:
        """Extract clean domain from website URL"""
        if not website.startswith(('http://', 'https://')):
            website = 'https://' + website
        return urlparse(website).netloc.lower()
    
    async def try_api_extraction(self, domain: str, api_name: str) -> Optional[bytes]:
        """Try to extract logo using specific API"""
        if not self.session:
            return None
            
        config = self.api_configs.get(api_name, {})
        url = config.get('url_template', '').format(domain=domain)
        headers = config.get('headers', {})
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                        data = await response.read()
                        if len(data) > 500:  # Minimum size check
                            return data
        except Exception:
            pass
        
        return None
    
    async def extract_logo_tiered(self, website: str, max_tier: int = 8) -> Dict:
        """Fast tiered logo extraction - only use proven fast APIs for speed"""
        domain = self.extract_domain(website)
        
        # Check cache first
        cached_result = self._load_from_cache(domain)
        if cached_result:
            return cached_result
        
        result = {
            'website': website,
            'domain': domain,
            'logo_data': None,
            'source': None,
            'size_bytes': 0,
            'success': False
        }
        
        # Use only fast, reliable APIs for initial extraction (for speed)
        fast_apis = ['clearbit', 'logo_dev', 'google_favicon', 'duckduckgo', 'iconhorse']
        
        for api_name in fast_apis:
            if api_name in self.api_configs:
                api_config = self.api_configs[api_name]
                if api_config.get('tier', 5) <= max_tier:
                    logo_data = await self.try_api_extraction(domain, api_name)
                    if logo_data:
                        result.update({
                            'logo_data': logo_data,
                            'source': api_name,
                            'size_bytes': len(logo_data),
                            'success': True
                        })
                        
                        # Save to cache
                        self._save_to_cache(domain, result)
                        return result
        
        return result
    
    async def extract_logo_all_apis(self, website: str) -> Dict:
        """Try ALL available APIs for maximum recovery chance"""
        domain = self.extract_domain(website)
        
        # Check cache first
        cached_result = self._load_from_cache(domain)
        if cached_result:
            return cached_result
        
        result = {
            'website': website,
            'domain': domain,
            'logo_data': None,
            'source': None,
            'size_bytes': 0,
            'success': False
        }
        
        # Try every single API we have
        for api_name, api_config in self.api_configs.items():
            logo_data = await self.try_api_extraction(domain, api_name)
            if logo_data:
                result.update({
                    'logo_data': logo_data,
                    'source': api_name,
                    'size_bytes': len(logo_data),
                    'success': True
                })
                
                # Save to cache
                self._save_to_cache(domain, result)
                return result
        
        return result
    
    async def batch_extract_logos_enhanced(self, websites: List[str], max_tier: int = 8, 
                                         batch_size: int = 50) -> List[Dict]:
        """Enhanced batch processing with progress tracking"""
        results = []
        total = len(websites)
        
        print(f"Starting enhanced logo extraction for {total} websites...")
        print(f"Target success rate: >98%")
        print(f"Using {len(self.api_configs)} API sources with {max_tier} tiers")
        
        for i in range(0, total, batch_size):
            batch = websites[i:i + batch_size]
            batch_start = time.time()
            
            # Process batch
            tasks = [self.extract_logo_tiered(website, max_tier) for website in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in batch_results if isinstance(r, dict)]
            results.extend(valid_results)
            
            # Progress tracking
            batch_time = time.time() - batch_start
            completed = min(i + batch_size, total)
            success_count = sum(1 for r in results if r.get('success', False))
            success_rate = (success_count / completed) * 100 if completed > 0 else 0
            
            print(f"Batch {i//batch_size + 1}: {completed}/{total} websites processed "
                  f"({success_rate:.1f}% success, {batch_time:.1f}s)")
            
            # Small delay between batches to be respectful
            await asyncio.sleep(0.5)
        
        return results

    async def parallel_batch_extract(self, websites: List[str], max_tier: int = 8, 
                                   batch_size: int = 30, num_workers: int = 4,
                                   delay_between_batches: float = 1.0) -> List[Dict]:
        """Parallel batch processing with multiple concurrent batches"""
        results_queue = Queue()
        lock = threading.Lock()
        
        # Split websites into chunks for parallel processing
        chunks = [websites[i:i + batch_size] for i in range(0, len(websites), batch_size)]
        total_batches = len(chunks)
        
        print(f"Starting PARALLEL logo extraction for {len(websites)} websites...")
        print(f"Using {num_workers} parallel workers processing {total_batches} batches")
        print(f"Batch size: {batch_size}, Delay between batches: {delay_between_batches}s")
        print(f"Target success rate: >98%")
        
        async def process_batch_with_delay(chunk, batch_id):
            """Process a single batch with random delay to avoid API blocking"""
            # Add random delay to stagger requests across workers
            initial_delay = random.uniform(0, delay_between_batches * 0.5)
            await asyncio.sleep(initial_delay)
            
            batch_start = time.time()
            
            # Process batch with smaller sub-batches to be more respectful
            sub_batch_size = min(10, len(chunk))
            batch_results = []
            
            for i in range(0, len(chunk), sub_batch_size):
                sub_chunk = chunk[i:i + sub_batch_size]
                tasks = [self.extract_logo_tiered(website, max_tier) for website in sub_chunk]
                sub_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions
                valid_results = [r for r in sub_results if isinstance(r, dict)]
                batch_results.extend(valid_results)
                
                # Small delay between sub-batches
                if i + sub_batch_size < len(chunk):
                    await asyncio.sleep(0.2)
            
            batch_time = time.time() - batch_start
            
            # Thread-safe progress reporting
            with lock:
                results_queue.put(batch_results)
                
                # Calculate current stats
                all_results = []
                temp_queue = Queue()
                while not results_queue.empty():
                    batch_data = results_queue.get()
                    all_results.extend(batch_data)
                    temp_queue.put(batch_data)
                
                # Restore queue
                while not temp_queue.empty():
                    results_queue.put(temp_queue.get())
                
                completed_websites = len(all_results)
                success_count = sum(1 for r in all_results if r.get('success', False))
                success_rate = (success_count / completed_websites) * 100 if completed_websites > 0 else 0
                
                print(f"Worker batch {batch_id}: {len(chunk)} websites processed "
                      f"({batch_time:.1f}s) | Total: {completed_websites}/{len(websites)} "
                      f"({success_rate:.1f}% success)")
        
        # Create async tasks for parallel batch processing
        async def run_parallel_batches():
            tasks = []
            for i, chunk in enumerate(chunks):
                # Stagger task creation to avoid overwhelming APIs
                if i > 0 and i % num_workers == 0:
                    await asyncio.sleep(delay_between_batches)
                
                task = asyncio.create_task(process_batch_with_delay(chunk, i + 1))
                tasks.append(task)
                
                # Limit concurrent tasks
                if len(tasks) >= num_workers:
                    # Wait for at least one task to complete
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)
            
            # Wait for remaining tasks
            if tasks:
                await asyncio.gather(*tasks)
        
        # Run parallel processing
        await run_parallel_batches()
        
        # Collect all results
        all_results = []
        while not results_queue.empty():
            batch_results = results_queue.get()
            all_results.extend(batch_results)
        
    async def fast_parallel_extract(self, websites: List[str], max_tier: int = 8,
                                   batch_size: int = 100, concurrent_batches: int = 3) -> List[Dict]:
        """Fast parallel processing - simple and efficient"""
        
        print(f"Starting FAST PARALLEL logo extraction for {len(websites)} websites...")
        print(f"Processing {concurrent_batches} batches concurrently")
        print(f"Batch size: {batch_size}")
        
        # Split into batches
        batches = [websites[i:i + batch_size] for i in range(0, len(websites), batch_size)]
        total_batches = len(batches)
        
        async def process_single_batch(batch, batch_id):
            """Process a single batch quickly"""
            batch_start = time.time()
            
            # Process entire batch concurrently (no artificial delays)
            tasks = [self.extract_logo_tiered(website, max_tier) for website in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in batch_results if isinstance(r, dict)]
            
            batch_time = time.time() - batch_start
            success_count = sum(1 for r in valid_results if r.get('success', False))
            success_rate = (success_count / len(valid_results)) * 100 if valid_results else 0
            
            print(f"Batch {batch_id}: {len(batch)} websites in {batch_time:.1f}s "
                  f"({success_rate:.1f}% success, {len(batch)/batch_time:.1f} sites/sec)")
            
            return valid_results
        
        # Process batches in groups of concurrent_batches
        all_results = []
        
        for i in range(0, total_batches, concurrent_batches):
            # Get next group of batches
            batch_group = batches[i:i + concurrent_batches]
            
            # Process this group concurrently
            group_tasks = [
                process_single_batch(batch, i + j + 1) 
                for j, batch in enumerate(batch_group)
            ]
            
            group_results = await asyncio.gather(*group_tasks)
            
            # Flatten results from this group
            for batch_results in group_results:
                all_results.extend(batch_results)
            
            # Brief pause between groups (not between individual requests)
            if i + concurrent_batches < total_batches:
                await asyncio.sleep(0.1)  # Very short pause
        
        return all_results
    
    async def try_dns_favicon_extraction(self, domain: str) -> Optional[bytes]:
        """Try direct DNS-based favicon extraction"""
        favicon_urls = [
            f"https://{domain}/favicon.ico",
            f"https://www.{domain}/favicon.ico",
            f"https://{domain}/apple-touch-icon.png",
            f"https://www.{domain}/apple-touch-icon.png",
            f"https://{domain}/favicon-32x32.png",
            f"https://www.{domain}/favicon-32x32.png",
            f"https://{domain}/favicon-16x16.png",
            f"https://www.{domain}/favicon-16x16.png",
            f"https://{domain}/apple-touch-icon-120x120.png",
            f"https://www.{domain}/apple-touch-icon-120x120.png"
        ]
        
        for url in favicon_urls:
            try:
                async with self.session.get(url, timeout=8) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        if any(img in content_type for img in ['image/', 'application/octet-stream']):
                            data_bytes = await response.read()
                            if len(data_bytes) > 300:  # Minimum valid favicon size
                                return data_bytes
            except Exception:
                continue
        
        return None

    async def extract_logo_with_dns_fallback(self, website: str) -> Dict:
        """Extract logo with smart API selection + DNS fallback for recovery"""
        domain = self.extract_domain(website)
        
        # Check cache first
        cached_result = self._load_from_cache(domain)
        if cached_result:
            return cached_result
        
        result = {
            'website': website,
            'domain': domain,
            'logo_data': None,
            'source': None,
            'size_bytes': 0,
            'success': False
        }
        
        # Smart API selection: try high-quality APIs first, then backup APIs
        priority_apis = ['clearbit', 'logo_dev', 'google_favicon', 'duckduckgo', 'iconhorse', 'besticon', 'favicon_io']
        
        # Try priority APIs first (fast and reliable)
        for api_name in priority_apis:
            if api_name in self.api_configs:
                logo_data = await self.try_api_extraction(domain, api_name)
                if logo_data:
                    result.update({
                        'logo_data': logo_data,
                        'source': api_name,
                        'size_bytes': len(logo_data),
                        'success': True
                    })
                    self._save_to_cache(domain, result)
                    return result
        
        # Try additional APIs from config (but limit to prevent slowdown)
        additional_apis = [name for name in self.api_configs.keys() if name not in priority_apis]
        for api_name in additional_apis[:10]:  # Limit to 10 additional APIs max
            logo_data = await self.try_api_extraction(domain, api_name)
            if logo_data:
                result.update({
                    'logo_data': logo_data,
                    'source': api_name,
                    'size_bytes': len(logo_data),
                    'success': True
                })
                self._save_to_cache(domain, result)
                return result
        
        # Final fallback: DNS favicon extraction
        logo_data = await self.try_dns_favicon_extraction(domain)
        if logo_data:
            result.update({
                'logo_data': logo_data,
                'source': 'dns_favicon',
                'size_bytes': len(logo_data),
                'success': True
            })
            self._save_to_cache(domain, result)
            return result
        
        return result

    async def parallel_recovery_extract_failed(self, failed_websites: List[str], 
                                             num_workers: int = 4) -> List[Dict]:
        """Parallel recovery extraction with DNS fallback and threading"""
        if not failed_websites:
            return []
        
        print(f"\n PARALLEL RECOVERY MODE: Extracting {len(failed_websites)} failed logos")
        print(f"Using {num_workers} parallel workers with DNS fallback")
        
        start_time = time.time()
        
        async def process_recovery_batch(batch, batch_id):
            """Process a recovery batch with full API + DNS extraction"""
            batch_start = time.time()
            
            # Use the enhanced extraction with DNS fallback
            tasks = [self.extract_logo_with_dns_fallback(website) for website in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in batch_results if isinstance(r, dict)]
            
            batch_time = time.time() - batch_start
            recovered_count = sum(1 for r in valid_results if r.get('success', False))
            recovery_rate = (recovered_count / len(valid_results)) * 100 if valid_results else 0
            
            print(f"  Recovery worker {batch_id}: {len(batch)} websites in {batch_time:.1f}s "
                  f"({recovery_rate:.1f}% recovered, {len(batch)/batch_time:.1f} sites/sec)")
            
            return valid_results
        
        # Split into batches for parallel processing
        batch_size = max(10, len(failed_websites) // (num_workers * 2))  # Smaller batches for recovery
        batches = [failed_websites[i:i + batch_size] for i in range(0, len(failed_websites), batch_size)]
        
        # Process batches in groups of num_workers
        all_results = []
        
        for i in range(0, len(batches), num_workers):
            batch_group = batches[i:i + num_workers]
            
            # Process this group in parallel
            group_tasks = [
                process_recovery_batch(batch, i + j + 1)
                for j, batch in enumerate(batch_group)
            ]
            
            group_results = await asyncio.gather(*group_tasks)
            
            # Flatten results
            for batch_results in group_results:
                all_results.extend(batch_results)
            
            # Brief pause between groups
            if i + num_workers < len(batches):
                await asyncio.sleep(0.5)
        
        total_time = time.time() - start_time
        final_recovered = sum(1 for r in all_results if r.get('success', False))
        
        print(f" Parallel recovery complete: {final_recovered}/{len(failed_websites)} logos recovered in {total_time:.1f}s")
        print(f"   Recovery rate: {(final_recovered/len(failed_websites)*100):.1f}%")
        
        return all_results

    async def recovery_extract_failed(self, failed_websites: List[str]) -> List[Dict]:
        """Legacy recovery method - now calls the enhanced parallel version"""
        return await self.parallel_recovery_extract_failed(failed_websites, num_workers=4)

async def comprehensive_logo_scraping_fast():
    """Run comprehensive logo scraping with fast parallel processing"""
    
    # Load domains from parquet
    print("Loading domains from parquet file...")
    df = pd.read_parquet('logos.snappy.parquet')
    websites = df['domain'].tolist()
    
    print(f"Loaded {len(websites)} domains for fast parallel scraping")
    
    # Initialize enhanced extractor
    async with EnhancedAPILogoExtractor() as extractor:
        
        # Run fast parallel extraction
        start_time = time.time()
        results = await extractor.fast_parallel_extract(
            websites, 
            max_tier=7,  # Use all 7 API tiers for maximum coverage
            batch_size=80,  # Larger batches for speed
            concurrent_batches=4  # Process 4 batches simultaneously
        )
        total_time = time.time() - start_time
        
        # Analyze results
        total_websites = len(websites)
        successful_extractions = [r for r in results if r.get('success', False)]
        success_count = len(successful_extractions)
        success_rate = (success_count / total_websites) * 100
        
        print(f"\n" + "="*80)
        print(f"FAST PARALLEL COMPREHENSIVE LOGO SCRAPING RESULTS")
        print(f"="*80)
        print(f"Total websites: {total_websites:,}")
        print(f"Successful extractions: {success_count:,}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Failed extractions: {total_websites - success_count:,}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per website: {total_time/total_websites:.3f}s")
        print(f"Processing speed: {total_websites/total_time:.1f} websites/second")
        
        # Source breakdown
        source_counts = {}
        total_bytes = 0
        for result in successful_extractions:
            source = result.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            total_bytes += result.get('size_bytes', 0)
        
        print(f"\nSource breakdown:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / success_count) * 100 if success_count > 0 else 0
            print(f"  {source}: {count:,} logos ({percentage:.1f}%)")
        
        print(f"\nData statistics:")
        print(f"  Total logo data: {total_bytes / (1024*1024):.1f} MB")
        if success_count > 0:
            print(f"  Average logo size: {total_bytes / success_count / 1024:.1f} KB")
        
        # Check if we achieved target, if not try enhanced recovery
        if success_rate >= 98.0:
            print(f"\n SUCCESS! Achieved target of >98% success rate ({success_rate:.2f}%)")
        else:
            print(f"\n  Initial success rate: {success_rate:.2f}% < 98.0% target")
            remaining_needed = int((98.0 * total_websites / 100) - success_count)
            print(f"   Need {remaining_needed} more successful extractions")
            
            # Attempt enhanced parallel recovery with DNS fallback
            failed_websites = [websites[i] for i, result in enumerate(results) 
                             if not result.get('success', False)]
            
            if failed_websites:
                print(f"\n Starting enhanced recovery for {len(failed_websites)} failed websites...")
                recovery_results = await extractor.parallel_recovery_extract_failed(
                    failed_websites, num_workers=6)  # More workers for recovery
                
                # Update results with recovered logos
                recovery_successful = [r for r in recovery_results if r.get('success', False)]
                if recovery_successful:
                    # Create mapping for quick lookup
                    recovery_dict = {r['website']: r for r in recovery_successful}
                    
                    # Replace failed results with successful recovery results
                    for i, result in enumerate(results):
                        if not result.get('success', False) and result['website'] in recovery_dict:
                            results[i] = recovery_dict[result['website']]
                    
                    # Recalculate all stats
                    successful_extractions = [r for r in results if r.get('success', False)]
                    success_count = len(successful_extractions)
                    success_rate = (success_count / total_websites) * 100
                    
                    # Recalculate source breakdown and data stats
                    source_counts = {}
                    total_bytes = 0
                    for result in successful_extractions:
                        source = result.get('source', 'unknown')
                        source_counts[source] = source_counts.get(source, 0) + 1
                        total_bytes += result.get('size_bytes', 0)
                    
                    print(f"\n ENHANCED RECOVERY RESULTS:")
                    print(f"   Recovered: {len(recovery_successful)} additional logos")
                    print(f"   Final success rate: {success_rate:.2f}%")
                    print(f"   Updated data size: {total_bytes / (1024*1024):.1f} MB")
                    
                    # Show recovery source breakdown
                    recovery_sources = {}
                    for r in recovery_successful:
                        source = r.get('source', 'unknown')
                        recovery_sources[source] = recovery_sources.get(source, 0) + 1
                    
                    if recovery_sources:
                        print(f"   Recovery sources:")
                        for source, count in sorted(recovery_sources.items(), key=lambda x: x[1], reverse=True):
                            print(f"     {source}: {count} logos")
                    
                    if success_rate >= 98.0:
                        print(f"    ENHANCED RECOVERY ACHIEVED >98% target!")
                    else:
                        still_needed = int((98.0 * total_websites / 100) - success_count)
                        print(f"   Still need {still_needed} more successful extractions")
                else:
                    print(f"     Recovery did not find additional logos")
        
        # Save comprehensive results (with recovery data)
        save_data = {
            'websites': websites,
            'logo_results': results,
            'successful_logos': successful_extractions,
            'metadata': {
                'total_websites': total_websites,
                'success_count': success_count,
                'success_rate': success_rate,
                'processing_time': total_time,
                'processing_speed': total_websites/total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_breakdown': source_counts,
                'method': 'fast_parallel_processing_with_enhanced_recovery'
            }
        }
        
        # Save to new file
        output_file = 'comprehensive_logo_extraction_fast_results.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\nResults saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        return save_data

async def comprehensive_logo_scraping_advanced():
    """Run comprehensive logo scraping with advanced parallel processing"""
    
    # Load domains from parquet
    print("Loading domains from parquet file...")
    df = pd.read_parquet('logos.snappy.parquet')
    websites = df['domain'].tolist()
    
    print(f"Loaded {len(websites)} domains for advanced parallel scraping")
    
    # Initialize enhanced extractor
    async with EnhancedAPILogoExtractor() as extractor:
        
        # Run advanced parallel extraction
        start_time = time.time()
        results = await extractor.advanced_parallel_extract(
            websites, 
            max_tier=7,  # Use all 7 API tiers for maximum coverage
            batch_size=15,  # Smaller batches for better control
            max_workers=3,  # Conservative worker count
            requests_per_second=8  # Rate limiting to avoid blocks
        )
        total_time = time.time() - start_time
        
        # Analyze results
        total_websites = len(websites)
        successful_extractions = [r for r in results if r.get('success', False)]
        success_count = len(successful_extractions)
        success_rate = (success_count / total_websites) * 100
        
        print(f"\n" + "="*80)
        print(f"ADVANCED PARALLEL COMPREHENSIVE LOGO SCRAPING RESULTS")
        print(f"="*80)
        print(f"Total websites: {total_websites:,}")
        print(f"Successful extractions: {success_count:,}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Failed extractions: {total_websites - success_count:,}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per website: {total_time/total_websites:.3f}s")
        print(f"Processing speed: {total_websites/total_time:.1f} websites/second")
        
        # Source breakdown
        source_counts = {}
        total_bytes = 0
        for result in successful_extractions:
            source = result.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            total_bytes += result.get('size_bytes', 0)
        
        print(f"\nSource breakdown:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / success_count) * 100 if success_count > 0 else 0
            print(f"  {source}: {count:,} logos ({percentage:.1f}%)")
        
        print(f"\nData statistics:")
        print(f"  Total logo data: {total_bytes / (1024*1024):.1f} MB")
        if success_count > 0:
            print(f"  Average logo size: {total_bytes / success_count / 1024:.1f} KB")
        
        # Check if we achieved target
        if success_rate >= 98.0:
            print(f"\n SUCCESS! Achieved target of >98% success rate ({success_rate:.2f}%)")
        else:
            print(f"\n  Need improvement: {success_rate:.2f}% < 98.0% target")
            remaining_needed = int((98.0 * total_websites / 100) - success_count)
            print(f"   Need {remaining_needed} more successful extractions")
        
        # Save comprehensive results
        save_data = {
            'websites': websites,
            'logo_results': results,
            'successful_logos': successful_extractions,
            'metadata': {
                'total_websites': total_websites,
                'success_count': success_count,
                'success_rate': success_rate,
                'processing_time': total_time,
                'processing_speed': total_websites/total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_breakdown': source_counts,
                'method': 'advanced_parallel_processing'
            }
        }
        
        # Save to new file
        output_file = 'comprehensive_logo_extraction_advanced_results.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\nResults saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        return save_data

async def comprehensive_logo_scraping_parallel():
    """Run comprehensive logo scraping with parallel batch processing to achieve >98% success rate"""
    
    # Load domains from parquet
    print("Loading domains from parquet file...")
    df = pd.read_parquet('logos.snappy.parquet')
    websites = df['domain'].tolist()
    
    print(f"Loaded {len(websites)} domains for comprehensive parallel scraping")
    
    # Initialize enhanced extractor
    async with EnhancedAPILogoExtractor() as extractor:
        
        # Run parallel extraction with optimized parameters to avoid API blocking
        start_time = time.time()
        results = await extractor.parallel_batch_extract(
            websites, 
            max_tier=7,  # Use all 7 API tiers for maximum coverage
            batch_size=25,  # Smaller batches to be more respectful
            num_workers=3,  # Limited concurrent workers to avoid rate limiting
            delay_between_batches=1.5  # Longer delays to prevent blocking
        )
        total_time = time.time() - start_time
        
        # Analyze results
        total_websites = len(websites)
        successful_extractions = [r for r in results if r.get('success', False)]
        success_count = len(successful_extractions)
        success_rate = (success_count / total_websites) * 100
        
        print(f"\n" + "="*80)
        print(f"PARALLEL COMPREHENSIVE LOGO SCRAPING RESULTS")
        print(f"="*80)
        print(f"Total websites: {total_websites:,}")
        print(f"Successful extractions: {success_count:,}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Failed extractions: {total_websites - success_count:,}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per website: {total_time/total_websites:.2f}s")
        print(f"Processing speed: {total_websites/total_time:.1f} websites/second")
        
        # Source breakdown
        source_counts = {}
        total_bytes = 0
        for result in successful_extractions:
            source = result.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            total_bytes += result.get('size_bytes', 0)
        
        print(f"\nSource breakdown:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / success_count) * 100 if success_count > 0 else 0
            print(f"  {source}: {count:,} logos ({percentage:.1f}%)")
        
        print(f"\nData statistics:")
        print(f"  Total logo data: {total_bytes / (1024*1024):.1f} MB")
        if success_count > 0:
            print(f"  Average logo size: {total_bytes / success_count / 1024:.1f} KB")
        
        # Check if we achieved target
        if success_rate >= 98.0:
            print(f"\n SUCCESS! Achieved target of >98% success rate ({success_rate:.2f}%)")
        else:
            print(f"\n  Need improvement: {success_rate:.2f}% < 98.0% target")
            remaining_needed = int((98.0 * total_websites / 100) - success_count)
            print(f"   Need {remaining_needed} more successful extractions")
        
        # Save comprehensive results
        save_data = {
            'websites': websites,
            'logo_results': results,
            'successful_logos': successful_extractions,
            'metadata': {
                'total_websites': total_websites,
                'success_count': success_count,
                'success_rate': success_rate,
                'processing_time': total_time,
                'processing_speed': total_websites/total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_breakdown': source_counts,
                'method': 'parallel_batch_processing'
            }
        }
        
        # Save to new file
        output_file = 'comprehensive_logo_extraction_parallel_results.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\nResults saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        return save_data

async def comprehensive_logo_scraping():
    """Run comprehensive logo scraping to achieve >98% success rate (original sequential version)"""
    
    # Load domains from parquet
    print("Loading domains from parquet file...")
    df = pd.read_parquet('logos.snappy.parquet')
    websites = df['domain'].tolist()
    
    print(f"Loaded {len(websites)} domains for comprehensive scraping")
    
    # Initialize enhanced extractor
    async with EnhancedAPILogoExtractor() as extractor:
        
        # Run enhanced extraction
        start_time = time.time()
        results = await extractor.batch_extract_logos_enhanced(
            websites, 
            max_tier=7,  # Use all 7 API tiers for maximum coverage
            batch_size=50
        )
        total_time = time.time() - start_time
        
        # Analyze results
        total_websites = len(websites)
        successful_extractions = [r for r in results if r.get('success', False)]
        success_count = len(successful_extractions)
        success_rate = (success_count / total_websites) * 100
        
        print(f"\n" + "="*80)
        print(f"COMPREHENSIVE LOGO SCRAPING RESULTS")
        print(f"="*80)
        print(f"Total websites: {total_websites:,}")
        print(f"Successful extractions: {success_count:,}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Failed extractions: {total_websites - success_count:,}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per website: {total_time/total_websites:.2f}s")
        
        # Source breakdown
        source_counts = {}
        total_bytes = 0
        for result in successful_extractions:
            source = result.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            total_bytes += result.get('size_bytes', 0)
        
        print(f"\nSource breakdown:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / success_count) * 100
            print(f"  {source}: {count:,} logos ({percentage:.1f}%)")
        
        print(f"\nData statistics:")
        print(f"  Total logo data: {total_bytes / (1024*1024):.1f} MB")
        print(f"  Average logo size: {total_bytes / success_count / 1024:.1f} KB")
        
        # Check if we achieved target
        if success_rate >= 98.0:
            print(f"\n SUCCESS! Achieved target of >98% success rate ({success_rate:.2f}%)")
        else:
            print(f"\n  Need improvement: {success_rate:.2f}% < 98.0% target")
            remaining_needed = int((98.0 * total_websites / 100) - success_count)
            print(f"   Need {remaining_needed} more successful extractions")
        
        # Save comprehensive results
        save_data = {
            'websites': websites,
            'logo_results': results,
            'successful_logos': successful_extractions,
            'metadata': {
                'total_websites': total_websites,
                'success_count': success_count,
                'success_rate': success_rate,
                'processing_time': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_breakdown': source_counts
            }
        }
        
        # Save to new file
        output_file = 'comprehensive_logo_extraction_results.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\nResults saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        return save_data

if __name__ == "__main__":
    import sys
    
    # Check command line arguments for processing method
    method = 'fast'  # Default to fast parallel processing
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--sequential':
            method = 'sequential'
        elif sys.argv[1] == '--parallel':
            method = 'parallel'
        elif sys.argv[1] == '--advanced':
            method = 'advanced'
        elif sys.argv[1] == '--fast':
            method = 'fast'
        else:
            print("Available options: --sequential, --parallel, --advanced, --fast")
            sys.exit(1)
    
    print(f"Running {method.upper()} comprehensive logo scraping...")
    print(f"Available options:")
    print(f"  --sequential: Original sequential batch processing")
    print(f"  --parallel: Parallel batch processing with threading")
    print(f"  --advanced: Advanced parallel processing with rate limiting")
    print(f"  --fast: Fast parallel processing optimized for speed (default)")
    print("")
    
    # Run the appropriate scraping method
    if method == 'sequential':
        results = asyncio.run(comprehensive_logo_scraping())
    elif method == 'parallel':
        results = asyncio.run(comprehensive_logo_scraping_parallel())
    elif method == 'advanced':
        results = asyncio.run(comprehensive_logo_scraping_advanced())
    else:  # fast
        results = asyncio.run(comprehensive_logo_scraping_fast())
    
    print(f"\n" + "="*80)
    print(f"SCRAPING COMPLETE - Ready for enhanced analysis!")
    print(f"Method used: {method.capitalize()} processing")
    print(f"="*80)
