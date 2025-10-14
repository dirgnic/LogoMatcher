# Logo Clustering Pipeline

**Fast, explainable logo similarity analysis without ML clustering**

Group websites by visually identical/near-identical logos using Fourier methods + Union-Find clustering. No k-means or DBSCAN — designed for billion-record scale with free compute.

## 🎯 **Challenge Requirements**

- **>97% logo extraction rate** from websites
- **Group websites** with similar/identical logos  
- **No ML clustering** (k-means, DBSCAN) — favor explainable methods
- **Scalable to billions** of records

## 🚀 **Our Solution**

### **Three Fourier-Based Similarity Methods**
1. **pHash (DCT)**: Near-duplicate fingerprint (Hamming ≤ 6)
2. **FFT Low-Frequency**: Global shape signature (Cosine ≥ 0.985)  
3. **Fourier-Mellin**: Rotation/scale invariant (Max cosine ≥ 0.995)

### **Union-Find Clustering** 
- **OR fusion rule**: Any similarity method can trigger grouping
- **Connected components** = logo groups (no predefined cluster count)
- **Transitive grouping**: If A~B and B~C, then A,B,C grouped

### **Fast Extraction Pipeline**
- **DOM heuristics**: JSON-LD → header/nav → logo hints → fallbacks
- **Async HTTP/2** with connection pooling and rate limiting
- **Edge assist**: Cloudflare Workers for caching
- **Playwright fallback** for JS-heavy sites (~3% of traffic)

## 🔧 **Quick Start**

### **Basic Usage**
```bash
# Install dependencies
pip install -r requirements.txt

# Run clustering pipeline
python logo_cluster.py websites.txt --output clusters.json --trace_unions

# Analyze results with explainability
python knn_probe.py clusters.json --tree_output tree.png

# View results
cat clusters.json | jq '.clusters[] | {size: .size, websites: .websites}'
```

### **Input Format** (`websites.txt`)
```
ebay.cn
ebayglobalshipping.com
aamcoanaheim.net  
aamcoconyersga.com
walmart.com
```

### **Output** (`clusters.json`)
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "size": 2,
      "websites": ["ebay.cn", "ebayglobalshipping.com"],
      "logo_urls": ["https://ebay.cn/logo.png", "https://ebayglobalshipping.com/logo.png"]
    }
  ],
  "extraction_rate": 98.5,
  "total_websites": 100
}
```

## 📊 **Scale Pipeline**

### **GitHub Actions (Free Parallel Processing)**
```bash
# Trigger batch processing
gh workflow run logo-clustering.yml \
  -f batch_size=1000 \
  -f total_batches=20 \
  -f input_file=https://your-bucket.com/websites.txt
```

### **Cloudflare Workers (Edge Caching)**
```bash
# Deploy edge worker
cd cloudflare-worker
wrangler deploy
```

## 📈 **Performance**
- **Single runner**: 500-1000 sites/minute
- **20 parallel batches**: 10,000-20,000 sites/minute  
- **Monthly capacity**: 420-840 million sites
- **Cost**: Nearly $0 using free tiers

### Input Options

The program automatically detects and uses website lists in this order of priority:

1. **Parquet file**: `logos.snappy.parquet` (looks for columns: website, url, domain, site)
2. **Text file**: `logos_list` (one website URL per line)
3. **Sample data**: Built-in sample websites for demonstration

### Running the Analysis

```bash
python main.py
```

### Output Files

- `logo_matching_results.json`: Detailed analysis results
- `logo_groups.csv`: Group information in CSV format

## Algorithm Approach

### 1. Logo Extraction Strategy
- **Multiple Selectors**: Uses CSS selectors targeting common logo patterns
  - `img[alt*="logo"]`, `img[src*="logo"]`, `img[class*="logo"]`
  - `.logo img`, `#logo img`, `header img`, `.navbar img`
  - Favicon links and brand images
- **Fallback Chain**: Tries multiple selectors per website
- **URL Normalization**: Handles relative URLs and protocol variations

### 2. Image Processing Pipeline  
- **Standardization**: Resize to 128x128, convert to grayscale
- **Noise Reduction**: Gaussian blur and normalization
- **Feature Extraction**: Multiple complementary approaches

### 3. Similarity Metrics (Non-ML Clustering)
- **Perceptual Hash (40%)**: 8x8 DCT-based hash with Hamming distance
- **Histogram Comparison (30%)**: Normalized correlation of pixel distributions  
- **Structural Similarity (30%)**: Template matching for shape/structure

### 4. Grouping Algorithm
- **Pairwise Comparison**: O(n²) comparison with early termination
- **Threshold-based**: Configurable similarity threshold (default: 0.85)
- **Union-Find Logic**: Prevents duplicate groupings

## Why This Approach Works

1. **Multiple Fallbacks**: Ensures high extraction rate across different website structures
2. **Complementary Metrics**: Each metric captures different aspects of visual similarity
3. **Perceptual Focus**: Emphasizes features humans use to recognize logos
4. **Noise Resistance**: Preprocessing handles variations in size, format, compression

## Sample Output

```
LOGO MATCHING ANALYSIS RESULTS
============================================================
Total websites analyzed: 15
Successful logo extractions: 15
Extraction rate: 100.0%
Number of groups found: 15

Group Distribution:
  Groups with 1 website(s): 15

Unique Logos: 15 websites

Analysis complete!
Check 'logo_matching_results.json' for detailed results
Check 'logo_groups.csv' for group information
```

## Configuration

You can modify the similarity threshold in `logo_matcher.py`:

```python
class LogoMatcher:
    def __init__(self):
        self.similarity_threshold = 0.85  # Adjust between 0.0 and 1.0
```

## Technical Details

### Image Processing
- **Format Support**: JPEG, PNG, GIF, WebP, SVG
- **Size Normalization**: All images resized to 128x128 for comparison
- **Color Space**: Converted to grayscale for consistent analysis
- **Noise Reduction**: Gaussian blur to handle compression artifacts

### Similarity Calculation
The combined similarity score uses three metrics:

```python
combined_score = (hash_similarity * 0.4 + 
                 hist_similarity * 0.3 + 
                 struct_similarity * 0.3)
```

### Performance Considerations
- **Rate Limiting**: 1-second delay between website requests
- **Timeout Handling**: 10-second timeout for web requests
- **Error Recovery**: Graceful handling of failed downloads
- **Memory Management**: Images processed individually

## Scaling Considerations

For large datasets (millions of websites):

- **Parallel Processing**: Can be parallelized per website
- **Caching**: Image hashes can be cached for repeat analyses  
- **Incremental Updates**: New websites can be compared against existing clusters
- **Memory Optimization**: Streaming processing for large datasets
- **Database Storage**: Store hashes and results in database for persistence

## Future Enhancements

- Add color histogram matching for brand color analysis
- Implement SIFT/ORB feature matching for complex logos
- Add logo detection/segmentation for pages with multiple images
- Implement approximate nearest neighbor search for O(log n) similarity queries
- Add support for animated logos (GIF analysis)
- Machine learning enhancement while maintaining core computer vision approach

## Troubleshooting

### Common Issues

1. **Low extraction rate**: Check if websites are accessible and have logos
2. **Import errors**: Run `pip install -r requirements.txt`
3. **Timeout errors**: Increase timeout in `download_image()` method
4. **Memory issues**: Process websites in smaller batches

### Dependencies

- Python 3.7+
- OpenCV for image processing
- BeautifulSoup for HTML parsing
- Pillow for image handling
- NumPy for numerical operations
- Pandas for data handling (parquet support)

## License

This project is provided for educational and research purposes.

## Contributing

Feel free to submit issues, feature requests, or improvements to enhance the logo matching capabilities.
