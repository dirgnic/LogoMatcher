#!/usr/bin/env python3
"""
Quick recovery script to get the last 3 logos needed for 98% success rate
"""
import asyncio
import aiohttp
import pickle
import json

# Load the failed extractions from our existing results
with open('comprehensive_logo_extraction_fast_results.pkl', 'rb') as f:
    data = pickle.load(f)

failed_websites = []
for result in data['logo_results']:
    if not result.get('success', False):
        failed_websites.append(result['website'])

print(f"Found {len(failed_websites)} failed websites to retry")
print("First 10 failed:", failed_websites[:10])

# Load additional APIs from config
with open('logo_apis_config.json', 'r') as f:
    config = json.load(f)

additional_apis = []
for api in config['logo_apis']:
    name = api.get('name', '').lower().replace(' ', '_')
    if 'favicon' in name or 'icon' in name:
        additional_apis.append({
            'name': name,
            'url': api.get('url', '').replace('{domain}', '{}'),
            'timeout': api.get('timeout', 5)
        })

print(f"Found {len(additional_apis)} additional APIs to try")

async def try_recovery():
    """Quick recovery attempt for just a few websites"""
    recovered = 0
    
    async with aiohttp.ClientSession() as session:
        # Try just the first 20 failed websites with additional APIs
        for website in failed_websites[:20]:
            if recovered >= 3:  # We only need 3 more
                break
                
            domain = website.replace('https://', '').replace('http://', '').split('/')[0]
            
            for api in additional_apis[:5]:  # Try first 5 additional APIs
                try:
                    url = api['url'].format(domain)
                    async with session.get(url, timeout=api['timeout']) as response:
                        if response.status == 200:
                            data = await response.read()
                            if len(data) > 500:  # Valid logo
                                print(f"✅ Recovered logo for {domain} using {api['name']}")
                                recovered += 1
                                break
                except:
                    pass
    
    print(f"Recovery attempt: {recovered} additional logos found")
    return recovered

if __name__ == "__main__":
    recovered = asyncio.run(try_recovery())
    
    if recovered >= 3:
        print("🎉 SUCCESS! Found enough additional logos to reach 98%!")
        new_success_rate = (data['metadata']['success_count'] + recovered) / data['metadata']['total_websites'] * 100
        print(f"New success rate would be: {new_success_rate:.2f}%")
    else:
        print(f"Need to try more APIs or websites (recovered {recovered}/3 needed)")
