#!/usr/bin/env python3
"""
Logo Clustering Pipeline: Fast extraction + Fourier similarity + Union-Find clustering

No ML clustering (k-means/DBSCAN). Uses explainable methods:
- DOM heuristics for extraction (JSON-LD → header/nav → fallbacks)
- Fourier-based similarity (pHash/FFT/Fourier-Mellin)
- Union-find for connected components

Usage:
    python logo_cluster.py websites.txt --output clusters.json --trace_unions
"""

import asyncio
import aiohttp
import numpy as np
import cv2
from PIL import Image
import json
import argparse
import time
import hashlib
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
import io
from pathlib import Path

# Fourier analysis
from scipy.fft import fft2, fftshift
from sklearn.metrics.pairwise import cosine_similarity


class UnionFind:
    """Union-Find with path compression for clustering"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.n_components -= 1
        return True
    
    def get_components(self) -> Dict[int, List[int]]:
        components = defaultdict(list)
        for i in range(len(self.parent)):
            components[self.find(i)].append(i)
        return dict(components)


class FourierLogoAnalyzer:
    """Fourier-based logo similarity without ML clustering"""
    
    def __init__(self, 
                 phash_threshold: int = 6,          # Hamming distance ≤ 6
                 fft_threshold: float = 0.985,      # Cosine ≥ 0.985
                 fmt_threshold: float = 0.995):     # Fourier-Mellin ≥ 0.995
        
        self.phash_threshold = phash_threshold
        self.fft_threshold = fft_threshold
        self.fmt_threshold = fmt_threshold
    
    def compute_phash(self, img: np.ndarray) -> str:
        """Perceptual hash using DCT (Fourier cousin) - near-duplicate fingerprint"""
        # Convert to grayscale and resize to 32x32 for DCT
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        
        # DCT (like 2D Fourier but with cosines)
        dct = cv2.dct(np.float32(resized))
        
        # Take top-left 8x8 (low frequencies)
        dct_low = dct[0:8, 0:8]
        
        # Binary hash from median comparison
        median = np.median(dct_low)
        binary = dct_low > median
        
        return ''.join(['1' if b else '0' for b in binary.flatten()])
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Hamming distance between pHashes"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def compute_fft_features(self, img: np.ndarray) -> np.ndarray:
        """FFT low-frequency vector - global shape signature"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        resized = cv2.resize(gray, (128, 128))
        
        # 2D FFT
        fft = fft2(resized)
        fft_shifted = fftshift(fft)
        
        # Log magnitude
        magnitude = np.abs(fft_shifted)
        log_magnitude = np.log(magnitude + 1e-8)
        
        # Central 32x32 block (low frequencies)
        center = 64
        crop_size = 16
        low_freq = log_magnitude[center-crop_size:center+crop_size, 
                                center-crop_size:center+crop_size]
        
        # Normalize and flatten
        features = low_freq.flatten()
        return features / (np.linalg.norm(features) + 1e-8)
    
    def compute_fourier_mellin_signature(self, img: np.ndarray) -> np.ndarray:
        """Fourier-Mellin θ-signature for rotation/scale invariance"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        resized = cv2.resize(gray, (128, 128))
        
        # FFT magnitude in log-polar
        fft = fft2(resized)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # Average over radius to get θ-signature
        center = 64
        theta_samples = 64
        radius_samples = 32
        theta_signature = np.zeros(theta_samples)
        
        for i, theta in enumerate(np.linspace(0, 2*np.pi, theta_samples, endpoint=False)):
            radial_sum = 0
            for r in np.linspace(1, center-1, radius_samples):
                x = int(center + r * np.cos(theta))
                y = int(center + r * np.sin(theta))
                if 0 <= x < 128 and 0 <= y < 128:
                    radial_sum += magnitude[y, x]
            theta_signature[i] = radial_sum
        
        return theta_signature / (np.linalg.norm(theta_signature) + 1e-8)
    
    def compare_fourier_mellin(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Max cosine over circular shifts (rotation invariance)"""
        n = len(sig1)
        # FFT-based circular correlation
        sig1_fft = np.fft.rfft(sig1, n=2*n)
        sig2_fft = np.fft.rfft(sig2[::-1], n=2*n)
        correlation = np.fft.irfft(sig1_fft * sig2_fft)
        return np.max(correlation)
    
    def extract_features(self, img: np.ndarray) -> Dict:
        """Extract all Fourier features"""
        return {
            'phash': self.compute_phash(img),
            'fft_features': self.compute_fft_features(img),
            'fmt_signature': self.compute_fourier_mellin_signature(img)
        }
    
    def are_similar(self, feat1: Dict, feat2: Dict) -> Tuple[bool, Dict, str]:
        """Check similarity with OR fusion rule"""
        # pHash (Hamming distance ≤ 6)
        phash_dist = self.hamming_distance(feat1['phash'], feat2['phash'])
        phash_match = phash_dist <= self.phash_threshold
        
        # FFT (cosine ≥ 0.985)
        fft_sim = cosine_similarity(
            feat1['fft_features'].reshape(1, -1),
            feat2['fft_features'].reshape(1, -1)
        )[0, 0]
        fft_match = fft_sim >= self.fft_threshold
        
        # Fourier-Mellin (max cosine ≥ 0.995)
        fmt_sim = self.compare_fourier_mellin(feat1['fmt_signature'], feat2['fmt_signature'])
        fmt_match = fmt_sim >= self.fmt_threshold
        
        # OR fusion rule
        overall_match = phash_match or fft_match or fmt_match
        
        # Determine basis for match
        basis = []
        if phash_match: basis.append(f"pHash({phash_dist})")
        if fft_match: basis.append(f"FFT({fft_sim:.3f})")
        if fmt_match: basis.append(f"FMT({fmt_sim:.3f})")
        
        metrics = {
            'phash_distance': phash_dist,
            'phash_match': phash_match,
            'fft_similarity': fft_sim,
            'fft_match': fft_match,
            'fmt_similarity': fmt_sim,
            'fmt_match': fmt_match,
            'overall_match': overall_match
        }
        
        return overall_match, metrics, " + ".join(basis)


class FastLogoExtractor:
    """Fast logo extraction: DOM heuristics → fallbacks"""
    
    def __init__(self):
        self.logo_patterns = re.compile(r'(logo|brand|site-logo|company-logo)', re.IGNORECASE)
        self.session = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=15, connect=10)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=4)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'FastLogoBot/1.0 (+https://research.veridion.com)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def extract_logo_candidates(self, html: str, base_url: str) -> List[Tuple[str, str]]:
        """DOM heuristics: JSON-LD → header/nav → links → fallbacks"""
        soup = BeautifulSoup(html, 'html.parser')
        candidates = []
        
        # 1. JSON-LD Organization.logo (highest priority)
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict) and item.get('@type') in ['Organization', 'Brand']:
                        logo = item.get('logo')
                        if isinstance(logo, str):
                            candidates.append(('json-ld', urljoin(base_url, logo)))
                        elif isinstance(logo, dict) and logo.get('url'):
                            candidates.append(('json-ld', urljoin(base_url, logo['url'])))
            except (json.JSONDecodeError, AttributeError):
                continue
        
        # 2. Header/nav img with logo hints
        for area in ['header', 'nav', '.navbar', '.header', '.site-header', '.masthead']:
            container = soup.select_one(area)
            if container:
                for img in container.find_all('img'):
                    src = img.get('src')
                    if src and self._is_logo_candidate(img, src):
                        candidates.append(('header-nav', urljoin(base_url, src)))
        
        # 3. <a href="/"> <img> (homepage links)
        for link in soup.find_all('a', href=re.compile(r'^(/|index|home)')):
            img = link.find('img')
            if img and img.get('src'):
                candidates.append(('homepage-link', urljoin(base_url, img['src'])))
        
        # 4. Images with logo indicators (id|class|alt ~ /logo|brand/)
        for img in soup.find_all('img'):
            src = img.get('src')
            if src and self._is_logo_candidate(img, src):
                candidates.append(('logo-hints', urljoin(base_url, src)))
        
        # 5. Apple touch icons (good fallback)
        for link in soup.find_all('link', rel=re.compile(r'apple-touch-icon')):
            href = link.get('href')
            if href:
                candidates.append(('apple-touch-icon', urljoin(base_url, href)))
        
        # 6. Favicon (last resort)
        for link in soup.find_all('link', rel=re.compile(r'icon')):
            href = link.get('href')
            if href and not href.endswith('.ico'):  # Skip .ico files
                candidates.append(('favicon', urljoin(base_url, href)))
        
        return candidates
    
    def _is_logo_candidate(self, img, src: str) -> bool:
        """Check if image likely contains logo"""
        attrs_text = ' '.join([
            img.get('id', ''),
            ' '.join(img.get('class', [])),
            img.get('alt', ''),
            src
        ])
        return bool(self.logo_patterns.search(attrs_text))
    
    async def download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image and convert to OpenCV format"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Convert to PIL and handle transparency
                    img = Image.open(io.BytesIO(content))
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Convert to OpenCV
                    img_array = np.array(img)
                    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        return None
    
    async def extract_logo(self, website_url: str) -> Dict:
        """Extract single logo with fallback chain"""
        clean_url = website_url if website_url.startswith(('http://', 'https://')) else f"https://{website_url}"
        
        result = {
            'website': website_url,
            'logo_found': False,
            'logo_url': None,
            'logo_data': None,
            'method': None,
            'error': None
        }
        
        try:
            # Fetch HTML
            async with self.session.get(clean_url) as response:
                if response.status != 200:
                    result['error'] = f"HTTP {response.status}"
                    return result
                
                html = await response.text()
            
            # Extract candidates
            candidates = self.extract_logo_candidates(html, clean_url)
            if not candidates:
                result['error'] = 'No candidates found'
                return result
            
            # Try candidates in priority order
            for method, logo_url in candidates:
                img_data = await self.download_image(logo_url)
                if img_data is not None and img_data.shape[0] > 16 and img_data.shape[1] > 16:
                    result.update({
                        'logo_found': True,
                        'logo_url': logo_url,
                        'logo_data': img_data,
                        'method': method
                    })
                    return result
            
            result['error'] = 'No valid images found'
            
        except Exception as e:
            result['error'] = str(e)
        
        return result


class LogoClusterer:
    """Main clustering pipeline: extract → analyze → union-find"""
    
    def __init__(self, 
                 phash_threshold: int = 6,
                 fft_threshold: float = 0.985,
                 fmt_threshold: float = 0.995,
                 trace_unions: bool = False):
        
        self.analyzer = FourierLogoAnalyzer(phash_threshold, fft_threshold, fmt_threshold)
        self.trace_unions = trace_unions
        self.union_trace = []
    
    async def cluster_websites(self, websites: List[str]) -> Dict:
        """Complete pipeline: extract → cluster → results"""
        print(f"🚀 Clustering {len(websites)} websites...")
        start_time = time.time()
        
        # Extract logos
        print("📥 Extracting logos...")
        async with FastLogoExtractor() as extractor:
            tasks = [extractor.extract_logo(url) for url in websites]
            logo_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid results
        valid_logos = []
        for result in logo_results:
            if isinstance(result, dict) and result['logo_found']:
                valid_logos.append(result)
        
        extraction_rate = len(valid_logos) / len(websites) * 100
        print(f"✅ Extracted {len(valid_logos)}/{len(websites)} logos ({extraction_rate:.1f}%)")
        
        if len(valid_logos) < 2:
            return self._empty_result(websites, logo_results, extraction_rate)
        
        # Compute features
        print("🔬 Computing Fourier features...")
        features = []
        for logo in valid_logos:
            feat = self.analyzer.extract_features(logo['logo_data'])
            features.append(feat)
        
        # Pre-grouping: exact pHash matches
        print("🔗 Pre-grouping exact pHash matches...")
        phash_groups = self._pregroup_exact_phash(features)
        
        # Build similarity graph
        print("📊 Building similarity graph...")
        n = len(valid_logos)
        uf = UnionFind(n)
        similarity_pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                is_similar, metrics, basis = self.analyzer.are_similar(features[i], features[j])
                
                if is_similar:
                    uf.union(i, j)
                    similarity_pairs.append({
                        'i': i, 'j': j,
                        'website_i': valid_logos[i]['website'],
                        'website_j': valid_logos[j]['website'],
                        'basis': basis,
                        **metrics
                    })
                    
                    if self.trace_unions:
                        self.union_trace.append({
                            'website_i': valid_logos[i]['website'],
                            'website_j': valid_logos[j]['website'],
                            'basis': basis,
                            'metrics': metrics
                        })
        
        # Get connected components
        components = uf.get_components()
        clusters = []
        
        for component_id, indices in components.items():
            if len(indices) == 1:
                continue  # Skip singletons
                
            cluster = {
                'cluster_id': len(clusters),
                'size': len(indices),
                'websites': [valid_logos[i]['website'] for i in indices],
                'logo_urls': [valid_logos[i]['logo_url'] for i in indices],
                'methods': [valid_logos[i]['method'] for i in indices]
            }
            clusters.append(cluster)
        
        # Sort by size
        clusters.sort(key=lambda x: x['size'], reverse=True)
        
        elapsed = time.time() - start_time
        print(f"🎯 Found {len(clusters)} clusters in {elapsed:.1f}s")
        
        return {
            'clusters': clusters,
            'extraction_rate': extraction_rate,
            'total_websites': len(websites),
            'logos_extracted': len(valid_logos),
            'similarity_pairs': similarity_pairs,
            'union_trace': self.union_trace if self.trace_unions else [],
            'phash_groups': phash_groups,
            'processing_time': elapsed
        }
    
    def _pregroup_exact_phash(self, features: List[Dict]) -> Dict:
        """Pre-group exact pHash matches"""
        phash_map = defaultdict(list)
        for i, feat in enumerate(features):
            phash_map[feat['phash']].append(i)
        
        groups = {phash: indices for phash, indices in phash_map.items() if len(indices) > 1}
        return groups
    
    def _empty_result(self, websites: List[str], logo_results: List, extraction_rate: float) -> Dict:
        """Return empty clustering result"""
        return {
            'clusters': [],
            'extraction_rate': extraction_rate,
            'total_websites': len(websites),
            'logos_extracted': 0,
            'similarity_pairs': [],
            'union_trace': [],
            'phash_groups': {},
            'processing_time': 0.0
        }


async def main():
    parser = argparse.ArgumentParser(description="Logo clustering pipeline")
    parser.add_argument('input', help='Input file with websites (one per line)')
    parser.add_argument('--output', default='clusters.json', help='Output JSON file')
    parser.add_argument('--csv', help='Also export clusters.csv')
    parser.add_argument('--trace_unions', action='store_true', help='Trace union operations')
    parser.add_argument('--phash_threshold', type=int, default=6, help='pHash Hamming threshold')
    parser.add_argument('--fft_threshold', type=float, default=0.985, help='FFT cosine threshold')
    parser.add_argument('--fmt_threshold', type=float, default=0.995, help='Fourier-Mellin threshold')
    
    args = parser.parse_args()
    
    # Load websites
    websites = []
    with open(args.input, 'r') as f:
        for line in f:
            website = line.strip()
            if website and not website.startswith('#'):
                websites.append(website)
    
    print(f"📋 Loaded {len(websites)} websites from {args.input}")
    
    # Run clustering
    clusterer = LogoClusterer(
        phash_threshold=args.phash_threshold,
        fft_threshold=args.fft_threshold,
        fmt_threshold=args.fmt_threshold,
        trace_unions=args.trace_unions
    )
    
    result = await clusterer.cluster_websites(websites)
    
    # Save JSON
    with open(args.output, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_result = json.loads(json.dumps(result, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
        json.dump(json_result, f, indent=2)
    
    print(f"💾 Saved results to {args.output}")
    
    # Optional CSV export
    if args.csv:
        import pandas as pd
        rows = []
        for cluster in result['clusters']:
            for website in cluster['websites']:
                rows.append({
                    'cluster_id': cluster['cluster_id'],
                    'cluster_size': cluster['size'],
                    'website': website
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(args.csv, index=False)
        print(f"💾 Saved CSV to {args.csv}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"   Extraction rate: {result['extraction_rate']:.1f}%")
    print(f"   Clusters found: {len(result['clusters'])}")
    print(f"   Largest cluster: {max([c['size'] for c in result['clusters']], default=0)} websites")
    
    if result['clusters']:
        print(f"\n🔗 TOP CLUSTERS:")
        for i, cluster in enumerate(result['clusters'][:5]):
            print(f"   {i+1}. {cluster['size']} websites: {', '.join(cluster['websites'][:3])}{'...' if cluster['size'] > 3 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
