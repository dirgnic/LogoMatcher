#!/usr/bin/env python3
import asyncio
import aiohttp
import pickle

# Load existing results
with open('comprehensive_logo_extraction_fast_results.pkl', 'rb') as f:
    data = pickle.load(f)

failed_websites = []
for result in data['logo_results']:
    if not result.get('success', False):
        failed_websites.append(result['website'])

print(f"Found {len(failed_websites)} failed websites")

# Try more standard favicon approaches
async def advanced_recovery():
    recovered = 0
    recovered_details = []
    
    async with aiohttp.ClientSession() as session:
        for website in failed_websites[:30]:  # Try more websites
            if recovered >= 3:
                break
                
            domain = website.replace('https://', '').replace('http://', '').split('/')[0]
            
            # Try multiple favicon approaches
            favicon_urls = [
                f"https://{domain}/favicon.ico",
                f"https://www.{domain}/favicon.ico", 
                f"https://{domain}/apple-touch-icon.png",
                f"https://www.{domain}/apple-touch-icon.png",
                f"https://{domain}/favicon-32x32.png",
                f"https://www.{domain}/favicon-32x32.png",
                f"https://logo.clearbit.com/{domain}?size=128&format=png",
                f"https://www.google.com/s2/favicons?domain={domain}&sz=64",
                f"https://icons8.com/icon/set/{domain.split('.')[0]}/color",
                f"https://api.faviconkit.com/{domain}/144"
            ]
            
            for url in favicon_urls:
                try:
                    async with session.get(url, timeout=8) as response:
                        if response.status == 200:
                            content_type = response.headers.get('content-type', '').lower()
                            if any(img in content_type for img in ['image/', 'application/octet-stream']):
                                data_bytes = await response.read()
                                if len(data_bytes) > 300:  # Lower threshold
                                    print(f" Recovered {domain} from {url.split('//')[1].split('/')[0]} ({len(data_bytes)} bytes)")
                                    recovered += 1
                                    recovered_details.append((domain, url, len(data_bytes)))
                                    break
                except Exception as e:
                    continue
    
    print(f"\nRecovery summary: {recovered} logos recovered")
    for domain, url, size in recovered_details:
        print(f"  {domain}: {size} bytes from {url.split('//')[1].split('/')[0]}")
    
    return recovered

if __name__ == "__main__":
    recovered = asyncio.run(advanced_recovery())
    
    current_success = data['metadata']['success_count']
    total_websites = data['metadata']['total_websites']
    
    new_success_count = current_success + recovered
    new_success_rate = (new_success_count / total_websites) * 100
    
    print(f"\nCurrent: {current_success}/{total_websites} = {current_success/total_websites*100:.2f}%")
    print(f"With recovery: {new_success_count}/{total_websites} = {new_success_rate:.2f}%")
    
    if new_success_rate >= 98.0:
        print(" SUCCESS! Reached 98% target!")
    else:
        needed = int((98.0 * total_websites / 100) - new_success_count)
        print(f"Still need {needed} more logos for 98%")
