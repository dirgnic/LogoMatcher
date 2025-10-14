#!/usr/bin/env python3
"""
Enhanced Lightning Pipeline with Expanded API Pool
🚀 Target: 97%+ logo extraction success rate
"""

import asyncio
import aiohttp
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import pyarrow.parquet as pq
import time
from collections import defaultdict
from typing import List, Dict, Optional
import warnings
import base64
import json
from urllib.parse import urlparse, urljoin
warnings.filterwarnings('ignore')

class EnhancedAPILogoExtractor:
    """Enhanced logo extraction with massive API pool for 97%+ success rate"""
    
    def __init__(self):
        self.session = None
        # Expanded API pool (ordered by speed/reliability)
        self.logo_apis = [
            # Tier 1: Premium/Fast APIs
            {
                'name': 'Clearbit',
                'url': 'https://logo.clearbit.com/{domain}',
                'params': {},
                'headers': {},
                'timeout': 3,
                'tier': 1
            },
            {
                'name': 'LogoAPI',
                'url': 'https://api.logo.dev/{domain}',
                'params': {},
                'headers': {},
                'timeout': 4,
                'tier': 1
            },
            {
                'name': 'BrandAPI',
                'url': 'https://logo.api.brand.io/{domain}',
                'params': {},
                'headers': {},
                'timeout': 4,
                'tier': 1
            },
            
            # Tier 2: Google Services
            {
                'name': 'Google Favicon',
                'url': 'https://www.google.com/s2/favicons',
                'params': {'domain': '{domain}', 'sz': '128'},
                'headers': {},
                'timeout': 2,
                'tier': 2
            },
            {
                'name': 'Google Favicon HD',
                'url': 'https://www.google.com/s2/favicons',
                'params': {'domain': '{domain}', 'sz': '256'},
                'headers': {},
                'timeout': 3,
                'tier': 2
            },
            {
                'name': 'Google PageSpeed',
                'url': 'https://www.googleapis.com/pagespeedonline/v5/runPagespeed',
                'params': {'url': 'https://{domain}', 'fields': 'lighthouseResult.audits.largest-contentful-paint-element'},
                'headers': {},
                'timeout': 5,
                'tier': 2
            },
            
            # Tier 3: Alternative Favicon Services
            {
                'name': 'Favicon.io',
                'url': 'https://favicons.githubusercontent.com/{domain}',
                'params': {},
                'headers': {},
                'timeout': 3,
                'tier': 3
            },
            {
                'name': 'Icons8',
                'url': 'https://img.icons8.com/color/128/{domain}',
                'params': {},
                'headers': {},
                'timeout': 4,
                'tier': 3
            },
            {
                'name': 'LogoSearch',
                'url': 'https://logo.search.com/api/logo/{domain}',
                'params': {},
                'headers': {},
                'timeout': 4,
                'tier': 3
            },
            {
                'name': 'CompanyLogo',
                'url': 'https://company-logo.com/api/logo/{domain}',
                'params': {},
                'headers': {},
                'timeout': 4,
                'tier': 3
            },
            
            # Tier 4: Directory/Social APIs
            {
                'name': 'Wikidata',
                'url': 'https://www.wikidata.org/wiki/Special:EntityData/{domain}.json',
                'params': {},
                'headers': {},
                'timeout': 5,
                'tier': 4
            },
            {
                'name': 'Wikipedia',
                'url': 'https://en.wikipedia.org/api/rest_v1/page/summary/{domain}',
                'params': {},
                'headers': {},
                'timeout': 5,
                'tier': 4
            },
            {
                'name': 'Crunchbase',
                'url': 'https://api.crunchbase.com/api/v4/entities/organizations/{domain}',
                'params': {},
                'headers': {},
                'timeout': 6,
                'tier': 4
            },
            
            # Tier 5: Web Archive & Meta
            {
                'name': 'Internet Archive',
                'url': 'https://web.archive.org/web/timemap/json?url={domain}',
                'params': {},
                'headers': {},
                'timeout': 6,
                'tier': 5
            },
            {
                'name': 'DuckDuckGo',
                'url': 'https://api.duckduckgo.com/',
                'params': {'q': '{domain} logo', 'format': 'json', 'image_type': 'photo'},
                'headers': {},
                'timeout': 5,
                'tier': 5
            },
            
            # Tier 6: Fallback scrapers (as backup)
            {
                'name': 'Direct Favicon',
                'url': 'https://{domain}/favicon.ico',
                'params': {},
                'headers': {},
                'timeout': 3,
                'tier': 6
            },
            {
                'name': 'Apple Touch Icon',
                'url': 'https://{domain}/apple-touch-icon.png',
                'params': {},
                'headers': {},
                'timeout': 3,
                'tier': 6
            },
            {
                'name': 'Android Icon',
                'url': 'https://{domain}/android-chrome-192x192.png',
                'params': {},
                'headers': {},
                'timeout': 3,
                'tier': 6
            },
            {
                'name': 'Site Logo',
                'url': 'https://{domain}/logo.png',
                'params': {},
                'headers': {},
                'timeout': 3,
                'tier': 6
            },
            {
                'name': 'Site Logo SVG',
                'url': 'https://{domain}/logo.svg',
                'params': {},
                'headers': {},
                'timeout': 3,
                'tier': 6
            },
        ]
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=15)
        connector = aiohttp.TCPConnector(limit=300, limit_per_host=100)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': 'LogoMatcher/2.0 Enhanced'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def clean_domain(self, website: str) -> str:
        """Extract clean domain from website URL"""
        if website.startswith(('http://', 'https://')):
            return urlparse(website).netloc
        return website
    
    async def try_api_service(self, api_config: dict, domain: str) -> Optional[Dict]:
        """Try a single API service for logo"""
        try:
            # Format URL
            if '{domain}' in api_config['url']:
                url = api_config['url'].format(domain=domain)
            else:
                url = api_config['url']
            
            # Format params
            params = {}
            for key, value in api_config.get('params', {}).items():
                if '{domain}' in str(value):
                    params[key] = value.format(domain=domain)
                else:
                    params[key] = value
            
            # Make request
            timeout = aiohttp.ClientTimeout(total=api_config['timeout'])
            async with self.session.get(
                url, 
                params=params,
                headers=api_config.get('headers', {}),
                timeout=timeout
            ) as response:
                
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    # Handle different response types
                    if 'image' in content_type:
                        content = await response.read()
                        if len(content) > 300:  # Minimum viable logo size
                            return {
                                'data': content,
                                'url': str(response.url),
                                'content_type': content_type,
                                'size': len(content)
                            }
                    
                    elif 'json' in content_type:
                        # Handle JSON responses (like PageSpeed, Wikipedia)
                        json_data = await response.json()
                        logo_url = self.extract_logo_from_json(json_data, api_config['name'])
                        if logo_url:
                            # Download the actual logo
                            logo_result = await self.download_logo_from_url(logo_url)
                            if logo_result:
                                return logo_result
                
        except Exception as e:
            # Silent fail for speed
            pass
        
        return None
    
    def extract_logo_from_json(self, json_data: dict, api_name: str) -> Optional[str]:
        """Extract logo URL from JSON API responses"""
        try:
            if api_name == 'Wikipedia':
                if 'thumbnail' in json_data and 'source' in json_data['thumbnail']:
                    return json_data['thumbnail']['source']
            
            elif api_name == 'DuckDuckGo':
                if 'Results' in json_data and json_data['Results']:
                    for result in json_data['Results']:
                        if 'Icon' in result and result['Icon'].get('URL'):
                            return result['Icon']['URL']
            
            elif api_name == 'Google PageSpeed':
                # Extract from PageSpeed insights
                if 'lighthouseResult' in json_data:
                    audits = json_data.get('lighthouseResult', {}).get('audits', {})
                    lcp = audits.get('largest-contentful-paint-element', {})
                    if 'details' in lcp and 'items' in lcp['details']:
                        for item in lcp['details']['items']:
                            if item.get('node', {}).get('nodeLabel', '').lower() in ['logo', 'img']:
                                return item.get('node', {}).get('src')
                        
        except Exception:
            pass
        
        return None
    
    async def download_logo_from_url(self, logo_url: str) -> Optional[Dict]:
        """Download logo from extracted URL"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with self.session.get(logo_url, timeout=timeout) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        content = await response.read()
                        if len(content) > 300:
                            return {
                                'data': content,
                                'url': logo_url,
                                'content_type': content_type,
                                'size': len(content)
                            }
        except Exception:
            pass
        return None
    
    async def extract_logo_tiered(self, website: str, max_tier: int = 6) -> Dict:
        """Extract logo using tiered API approach for maximum success"""
        domain = self.clean_domain(website)
        
        result = {
            'website': website,
            'domain': domain,
            'logo_found': False,
            'logo_url': None,
            'logo_data': None,
            'method': 'enhanced_api',
            'api_service': None,
            'tier_used': None,
            'attempts': 0,
            'error': None
        }
        
        # Try APIs by tier for maximum efficiency
        for tier in range(1, max_tier + 1):
            tier_apis = [api for api in self.logo_apis if api.get('tier') == tier]
            
            # Try all APIs in current tier concurrently
            if tier_apis:
                tasks = [self.try_api_service(api_config, domain) for api_config in tier_apis]
                tier_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for success in this tier
                for i, logo_result in enumerate(tier_results):
                    if isinstance(logo_result, dict) and logo_result:
                        result.update({
                            'logo_found': True,
                            'logo_url': logo_result['url'],
                            'logo_data': logo_result['data'],
                            'method': 'enhanced_api',
                            'api_service': tier_apis[i]['name'],
                            'tier_used': tier,
                            'attempts': result['attempts'] + len(tier_apis)
                        })
                        return result
                
                result['attempts'] += len(tier_apis)
                
                # If we found something in tier 1-2, don't continue to slower tiers
                if tier <= 2:
                    await asyncio.sleep(0.1)  # Brief pause between tiers
        
        result['error'] = f'All {result["attempts"]} APIs failed'
        return result
    
    async def batch_extract_logos_enhanced(self, websites: List[str], max_tier: int = 6) -> List[Dict]:
        """Enhanced batch extraction with tier limits for speed vs coverage balance"""
        print(f"🚀 Enhanced API extraction: {len(websites)} websites (max tier: {max_tier})")
        start_time = time.time()
        
        # Process websites in smaller batches to prevent overwhelming
        batch_size = 50
        all_results = []
        
        for i in range(0, len(websites), batch_size):
            batch = websites[i:i + batch_size]
            print(f"   📦 Processing batch {i//batch_size + 1}/{(len(websites)-1)//batch_size + 1} ({len(batch)} websites)")
            
            # Process batch concurrently
            tasks = [self.extract_logo_tiered(website, max_tier) for website in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter results
            for j, result in enumerate(batch_results):
                if isinstance(result, dict):
                    all_results.append(result)
                else:
                    all_results.append({
                        'website': batch[j],
                        'logo_found': False,
                        'error': f'Exception: {type(result).__name__}'
                    })
            
            # Brief pause between batches
            await asyncio.sleep(0.2)
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in all_results if r['logo_found'])
        
        print(f"✅ Enhanced results: {successful}/{len(websites)} in {elapsed:.1f}s ({len(websites)/elapsed:.1f}/s)")
        print(f"🎯 Success rate: {successful/len(websites)*100:.1f}%")
        
        # Show tier and API breakdown
        tier_breakdown = defaultdict(int)
        api_breakdown = defaultdict(int)
        
        for result in all_results:
            if result['logo_found']:
                tier = result.get('tier_used', 'unknown')
                service = result.get('api_service', 'unknown')
                tier_breakdown[f"Tier {tier}"] += 1
                api_breakdown[service] += 1
        
        print("📊 Tier performance:")
        for tier, count in sorted(tier_breakdown.items()):
            print(f"   - {tier}: {count} logos")
        
        print("📊 Top API services:")
        for service, count in sorted(api_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {service}: {count}")
        
        return all_results


class LightningParquetProcessor:
    """Optimized parquet processing for 4000+ websites"""
    
    @staticmethod
    def load_parquet_fast(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load parquet with PyArrow for maximum speed"""
        print(f"⚡ Loading parquet: {file_path}")
        start_time = time.time()
        
        # Use PyArrow for fastest loading
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"📊 Sampled {sample_size} from {len(table)} total websites")
        
        elapsed = time.time() - start_time
        print(f"✅ Loaded {len(df)} websites in {elapsed:.2f}s")
        
        return df
    
    @staticmethod
    def get_website_column(df: pd.DataFrame) -> str:
        """Auto-detect website column"""
        website_cols = ['website', 'url', 'domain', 'site', 'link']
        for col in website_cols:
            if col in df.columns:
                return col
        
        # Check for columns containing 'web' or 'url'
        for col in df.columns:
            if any(term in col.lower() for term in ['web', 'url', 'domain']):
                return col
        
        # Default to first column
        return df.columns[0]


async def process_enhanced_lightning_pipeline(sample_size: int = 100, max_tier: int = 4):
    """Enhanced lightning-fast pipeline targeting 97%+ success rate"""
    
    print("🚀 ENHANCED LIGHTNING PIPELINE - TARGET 97%+ SUCCESS")
    print("=" * 65)
    
    # Step 1: Load parquet data
    df = LightningParquetProcessor.load_parquet_fast(
        'logos.snappy.parquet',
        sample_size=sample_size
    )
    
    # Get website column
    website_col = LightningParquetProcessor.get_website_column(df)
    print(f"📊 Website column detected: '{website_col}'")
    
    websites = df[website_col].dropna().tolist()
    print(f"📝 Processing {len(websites)} websites")
    
    # Step 2: Enhanced logo extraction
    print(f"\n🎯 ENHANCED LOGO EXTRACTION")
    print("-" * 35)
    print(f"🔧 Using API tiers 1-{max_tier} (speed vs coverage balance)")
    
    async with EnhancedAPILogoExtractor() as extractor:
        logo_results = await extractor.batch_extract_logos_enhanced(websites, max_tier=max_tier)
    
    # Step 3: Results analysis
    successful_logos = [r for r in logo_results if r['logo_found']]
    success_rate = len(successful_logos)/len(websites)*100
    
    print(f"\n✅ Enhanced extraction complete: {len(successful_logos)}/{len(websites)} logos")
    print(f"🎯 Success rate: {success_rate:.1f}%")
    
    # Recommendations for improvement
    if success_rate < 95:
        print(f"\n💡 To reach 97%+ success rate:")
        print(f"   - Increase max_tier to 6 (adds web scraping fallbacks)")
        print(f"   - Current tier limit: {max_tier}")
        if max_tier < 6:
            print(f"   - Try: max_tier=6 for maximum coverage")
    else:
        print(f"\n🎉 EXCELLENT! {success_rate:.1f}% success rate achieved!")
    
    # Step 4: Performance summary
    print(f"\n🎉 ENHANCED PIPELINE COMPLETE!")
    print(f"   - Websites processed: {len(websites)}")
    print(f"   - Logos extracted: {len(successful_logos)}")
    print(f"   - Success rate: {success_rate:.1f}%")
    print(f"   - API tier limit: {max_tier}")
    
    return {
        'websites': websites,
        'logo_results': logo_results,
        'successful_logos': successful_logos,
        'success_rate': success_rate
    }


if __name__ == "__main__":
    print("🚀 Starting Enhanced Lightning-Fast Logo Pipeline")
    print("💡 Target: 97%+ logo extraction success rate")
    
    # Test with different tier limits
    print("\n🔬 Testing tier performance...")
    
    # Quick test (tiers 1-2 only - fastest)
    print("\n--- TIER 1-2 TEST (Fastest) ---")
    results_fast = asyncio.run(process_enhanced_lightning_pipeline(sample_size=50, max_tier=2))
    
    # Balanced test (tiers 1-4 - good speed/coverage balance)
    print("\n--- TIER 1-4 TEST (Balanced) ---")
    results_balanced = asyncio.run(process_enhanced_lightning_pipeline(sample_size=50, max_tier=4))
    
    # Full coverage test (all tiers - maximum success rate)
    print("\n--- TIER 1-6 TEST (Maximum Coverage) ---")
    results_full = asyncio.run(process_enhanced_lightning_pipeline(sample_size=50, max_tier=6))
    
    print(f"\n🎯 TIER COMPARISON:")
    print(f"   - Tiers 1-2: {results_fast['success_rate']:.1f}% success")
    print(f"   - Tiers 1-4: {results_balanced['success_rate']:.1f}% success")  
    print(f"   - Tiers 1-6: {results_full['success_rate']:.1f}% success")
    
    print(f"\n💡 RECOMMENDATION:")
    if results_full['success_rate'] >= 97:
        print(f"   ✅ Use max_tier=6 for 97%+ success rate!")
    elif results_balanced['success_rate'] >= 95:
        print(f"   ⚡ Use max_tier=4 for good balance (95%+ success)")
    else:
        print(f"   🔧 Consider adding more API services for better coverage")
