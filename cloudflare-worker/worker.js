/**
 * Cloudflare Worker: Fast Logo URL Extraction
 * 
 * Edge assistance for logo clustering pipeline:
 * - Cache JSON-LD Organization.logo data
 * - Return logo URL candidates quickly  
 * - Reduce main pipeline load
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // Handle CORS
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }
    
    if (url.pathname === '/extract-logo-urls') {
      return handleLogoExtraction(request, env);
    }
    
    return new Response('Logo Extractor Edge Worker', { 
      headers: { 'Access-Control-Allow-Origin': '*' } 
    });
  },
};

async function handleLogoExtraction(request, env) {
  try {
    const { website_url } = await request.json();
    
    if (!website_url) {
      return jsonResponse({ error: 'website_url required' }, 400);
    }
    
    // Check cache first
    const cacheKey = `logo-urls:${website_url}`;
    const cached = await env.LOGO_CACHE.get(cacheKey);
    if (cached) {
      const data = JSON.parse(cached);
      data.cached = true;
      return jsonResponse(data);
    }
    
    // Extract logo URLs
    const result = await extractLogoUrls(website_url);
    
    // Cache for 24 hours
    await env.LOGO_CACHE.put(cacheKey, JSON.stringify(result), {
      expirationTtl: 86400
    });
    
    result.cached = false;
    return jsonResponse(result);
    
  } catch (error) {
    return jsonResponse({ 
      error: error.message,
      website_url: null,
      logo_candidates: []
    }, 500);
  }
}

async function extractLogoUrls(websiteUrl) {
  // Clean URL
  const cleanUrl = websiteUrl.startsWith('http') ? websiteUrl : `https://${websiteUrl}`;
  
  try {
    // Fetch HTML with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout
    
    const response = await fetch(cleanUrl, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'LogoExtractorBot/1.0 (+https://research.veridion.com)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      }
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const html = await response.text();
    const candidates = extractLogoCandidatesFromHtml(html, cleanUrl);
    
    return {
      website_url: websiteUrl,
      status: 'success',
      logo_candidates: candidates,
      extraction_time: new Date().toISOString()
    };
    
  } catch (error) {
    return {
      website_url: websiteUrl,
      status: 'error',
      error: error.message,
      logo_candidates: []
    };
  }
}

function extractLogoCandidatesFromHtml(html, baseUrl) {
  const candidates = [];
  
  // 1. JSON-LD Organization.logo (highest priority)
  const jsonLdMatches = html.match(/<script[^>]*type=["']application\/ld\+json["'][^>]*>(.*?)<\/script>/gis);
  if (jsonLdMatches) {
    for (const match of jsonLdMatches) {
      try {
        const scriptContent = match.replace(/<script[^>]*>/i, '').replace(/<\/script>/i, '');
        const data = JSON.parse(scriptContent);
        const items = Array.isArray(data) ? data : [data];
        
        for (const item of items) {
          if (item['@type'] === 'Organization' || item['@type'] === 'Brand') {
            const logo = item.logo;
            if (typeof logo === 'string') {
              candidates.push({
                method: 'json-ld',
                url: resolveUrl(logo, baseUrl),
                priority: 1
              });
            } else if (logo && logo.url) {
              candidates.push({
                method: 'json-ld',
                url: resolveUrl(logo.url, baseUrl),
                priority: 1
              });
            }
          }
        }
      } catch (e) {
        // Skip invalid JSON-LD
      }
    }
  }
  
  // 2. Header/nav images with logo patterns
  const logoPatterns = /(?:logo|brand|site-logo|company-logo)/i;
  const headerNavMatches = html.match(/<(?:header|nav)[^>]*>.*?<\/(?:header|nav)>/gis);
  if (headerNavMatches) {
    for (const headerNav of headerNavMatches) {
      const imgMatches = headerNav.match(/<img[^>]*>/gi);
      if (imgMatches) {
        for (const img of imgMatches) {
          const src = extractAttribute(img, 'src');
          if (src && logoPatterns.test(img)) {
            candidates.push({
              method: 'header-nav',
              url: resolveUrl(src, baseUrl),
              priority: 2
            });
          }
        }
      }
    }
  }
  
  // 3. Homepage link images
  const homeLinkMatches = html.match(/<a[^>]*href=["'](?:\/|index|home)[^"']*["'][^>]*>.*?<\/a>/gis);
  if (homeLinkMatches) {
    for (const link of homeLinkMatches) {
      const imgMatch = link.match(/<img[^>]*src=["']([^"']+)["'][^>]*>/i);
      if (imgMatch) {
        candidates.push({
          method: 'homepage-link',
          url: resolveUrl(imgMatch[1], baseUrl),
          priority: 3
        });
      }
    }
  }
  
  // 4. Apple touch icons
  const appleTouchIconMatches = html.match(/<link[^>]*rel=["'][^"']*apple-touch-icon[^"']*["'][^>]*>/gi);
  if (appleTouchIconMatches) {
    for (const link of appleTouchIconMatches) {
      const href = extractAttribute(link, 'href');
      if (href) {
        candidates.push({
          method: 'apple-touch-icon',
          url: resolveUrl(href, baseUrl),
          priority: 4
        });
      }
    }
  }
  
  // 5. Favicon (last resort, skip .ico files)
  const faviconMatches = html.match(/<link[^>]*rel=["'][^"']*icon[^"']*["'][^>]*>/gi);
  if (faviconMatches) {
    for (const link of faviconMatches) {
      const href = extractAttribute(link, 'href');
      if (href && !href.endsWith('.ico')) {
        candidates.push({
          method: 'favicon',
          url: resolveUrl(href, baseUrl),
          priority: 5
        });
      }
    }
  }
  
  // Sort by priority and remove duplicates
  const uniqueUrls = new Set();
  return candidates
    .filter(candidate => {
      if (uniqueUrls.has(candidate.url)) {
        return false;
      }
      uniqueUrls.add(candidate.url);
      return true;
    })
    .sort((a, b) => a.priority - b.priority)
    .slice(0, 10); // Limit to top 10 candidates
}

function extractAttribute(tag, attribute) {
  const match = tag.match(new RegExp(`${attribute}=["']([^"']+)["']`, 'i'));
  return match ? match[1] : null;
}

function resolveUrl(url, baseUrl) {
  try {
    return new URL(url, baseUrl).href;
  } catch {
    return url;
  }
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
  });
}
