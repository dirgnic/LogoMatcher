# Logo Clustering Pipeline: One-Page Overview

## 🎯 **Problem: Group Websites by Visually Identical/Near-Identical Logos**

**Goal**: Detect brand re-use across websites (not semantic similarity)  
**Constraints**: No ML clustering (k-means/DBSCAN) — favor fast, explainable methods  
**Scale**: Billions of websites with free/cheap compute  

---

## 🚀 **Pipeline Architecture**

```
websites.txt → DOM extraction → Fourier analysis → Union-Find → clusters.json
              ↓                 ↓                  ↓
           📥 Fast Path      🔬 3 Methods      🔗 Graph Components
           • JSON-LD         • pHash (DCT)      • Hamming ≤ 6
           • Header/nav      • FFT low-freq     • Cosine ≥ 0.985  
           • Logo hints      • Fourier-Mellin   • Max cosine ≥ 0.995
           • Fallbacks       • OR fusion rule   • Connected = grouped
```

---

## 📊 **Three Fourier Methods (No ML Clustering)**

| Method | Purpose | Threshold | Invariance |
|--------|---------|-----------|------------|
| **pHash (DCT)** | Near-duplicate fingerprint | Hamming ≤ 6 | Minor variations |
| **FFT Low-Freq** | Global shape signature | Cosine ≥ 0.985 | Size, position |
| **Fourier-Mellin** | Advanced matching | Max cosine ≥ 0.995 | Rotation, scale |

**OR Fusion Rule**: If any method says "match" → add edge to similarity graph

---

## 🔍 **Fast Extraction Strategy**

### **DOM Heuristics (Fast Path)**
1. **JSON-LD** `Organization.logo` (highest priority)
2. **Header/nav** `img` with logo class/id/alt patterns  
3. **Homepage links** `<a href="/"> <img>`
4. **Logo indicators** `id|class|alt` ~ `/logo|brand/`
5. **Fallbacks**: `apple-touch-icon` → cautious `og:image` → `favicon`

### **Rendered Fallback (Rare)**
- **Playwright** for JS-heavy pages (only when DOM fails)
- Detect `<img>` and CSS background logos in header
- ~3% of sites need this

---

## 🔗 **Union-Find Clustering (No k-means/DBSCAN)**

```python
# Build similarity graph
for each pair (logo_i, logo_j):
    if pHash_similar OR fft_similar OR fmt_similar:
        union_find.union(i, j)

# Get connected components = logo groups  
clusters = union_find.get_components()
```

**Benefits**: Transitive grouping, no predefined cluster count, O(n α(n)) complexity

---

## 🎯 **Explainability Features**

### **k-NN Probe (No Clustering)**
- **Interpretable features**: aspect ratio, RGB means, HSV hue bins, edge density, sharpness
- **Decision Tree** training to reveal cluster split rules  
- **Per-cluster profiles**: z-scores, distinguishing features

### **Example Tree Rule**
```
if aspect_ratio <= 1.2 and hue_bin_0 >= 0.3:
    → Cluster A (square, red-dominant logos)
else if edge_density >= 0.15:
    → Cluster B (high-detail logos)  
```

---

## ⚡ **Production Scale Pipeline**

### **Free/Cheap Compute Stack**
- **Edge**: Cloudflare Workers + KV (cache JSON-LD, icons)
- **Batch**: GitHub Actions matrix (10-20 shards, HTTP/2 pooling)  
- **Storage**: Neon/Supabase Postgres + Backblaze B2
- **Fallback**: Playwright queue for failed extractions

### **Performance Estimates**
- **Single runner**: 500-1000 sites/minute
- **20 parallel**: 10,000-20,000 sites/minute  
- **Monthly capacity**: 420-840 million sites
- **Cost**: Nearly $0 using free tiers

---

## 📦 **Deliverables**

| File | Purpose |
|------|---------|
| `logo_cluster.py` | Main pipeline: extract → analyze → cluster |
| `knn_probe.py` | Explainability: k-NN + Decision Tree rules |
| `clusters.json` | Results: grouped websites with similarity metrics |
| `clusters.csv` | Tabular export for analysis |
| `union_trace.json` | Debug: which pairs matched and why |

---

## 🔧 **Quick Start**

```bash
# Install dependencies
pip install -r requirements.txt

# Run clustering
python logo_cluster.py websites.txt --output clusters.json --trace_unions

# Analyze clusters  
python knn_probe.py clusters.json --tree_output tree.png

# Results
cat clusters.json | jq '.clusters[] | {size: .size, websites: .websites}'
```

---

## ✅ **Key Innovations**

1. **No ML clustering** — uses graph connectivity instead
2. **Fourier everywhere** — DCT, FFT, Fourier-Mellin for robustness
3. **OR fusion rule** — multiple similarity channels for recall  
4. **Explainable features** — human-interpretable decision rules
5. **Free-tier scaling** — GitHub Actions + edge compute architecture

**Result**: Fast, explainable logo grouping at billion-record scale without traditional ML clustering
