#!/usr/bin/env python3
"""
Save and Access Lightning Pipeline Results
🎯 Process and save your logo extraction results
"""

import asyncio
import pandas as pd
import json
import pickle
from lightning_pipeline import process_lightning_fast_pipeline
import time

async def save_results():
    """Run pipeline and save results to files"""
    print("🚀 Running pipeline and saving results...")
    
    # Run the full pipeline
    results = await process_lightning_fast_pipeline(sample_size=None)
    
    # Extract data
    websites = results['websites']
    logo_results = results['logo_results']
    successful_logos = results['successful_logos']
    
    print(f"\n💾 Saving results to files...")
    
    # 1. Save as JSON (human readable)
    results_json = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_websites': len(websites),
        'successful_extractions': len(successful_logos),
        'success_rate': len(successful_logos) / len(websites) * 100,
        'logo_results': logo_results,
        'summary': {
            'clearbit_logos': len([r for r in logo_results if r.get('api_service') == 'Clearbit']),
            'google_favicon_logos': len([r for r in logo_results if r.get('api_service') == 'Google Favicon']),
            'failed_extractions': len([r for r in logo_results if not r.get('logo_found', False)])
        }
    }
    
    with open('logo_extraction_results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    # 2. Save as CSV (easy to analyze)
    df_results = pd.DataFrame(logo_results)
    df_results.to_csv('logo_extraction_results.csv', index=False)
    
    # 3. Save as pickle (Python objects)
    with open('logo_extraction_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 4. Save successful logos separately
    df_successful = pd.DataFrame(successful_logos)
    df_successful.to_csv('successful_logo_extractions.csv', index=False)
    
    print(f"✅ Results saved to:")
    print(f"   📄 logo_extraction_results.json - Full results (human readable)")
    print(f"   📊 logo_extraction_results.csv - All results as CSV")
    print(f"   ✅ successful_logo_extractions.csv - Only successful extractions")
    print(f"   🐍 logo_extraction_results.pkl - Python objects")
    
    return results

def load_results():
    """Load previously saved results"""
    try:
        with open('logo_extraction_results.pkl', 'rb') as f:
            results = pickle.load(f)
        print("✅ Loaded results from logo_extraction_results.pkl")
        return results
    except FileNotFoundError:
        print("❌ No saved results found. Run save_results() first.")
        return None

def show_summary():
    """Show a quick summary of results"""
    try:
        with open('logo_extraction_results.json', 'r') as f:
            data = json.load(f)
        
        print("📊 LOGO EXTRACTION SUMMARY")
        print("=" * 50)
        print(f"🕒 Timestamp: {data['timestamp']}")
        print(f"🌐 Total websites: {data['total_websites']:,}")
        print(f"✅ Successful extractions: {data['successful_extractions']:,}")
        print(f"📈 Success rate: {data['success_rate']:.1f}%")
        print(f"\n🔧 API Service Breakdown:")
        print(f"   🏢 Clearbit: {data['summary']['clearbit_logos']:,}")
        print(f"   🌐 Google Favicon: {data['summary']['google_favicon_logos']:,}")
        print(f"   ❌ Failed: {data['summary']['failed_extractions']:,}")
        
    except FileNotFoundError:
        print("❌ No results file found. Run the pipeline first.")

def access_logos():
    """Access the logo data for further processing"""
    results = load_results()
    if results:
        successful_logos = results['successful_logos']
        print(f"🎯 You have {len(successful_logos)} logos ready for similarity analysis!")
        
        # Show first few examples
        print(f"\n📋 First 5 successful extractions:")
        for i, logo in enumerate(successful_logos[:5]):
            print(f"   {i+1}. {logo['website']} -> {logo['api_service']}")
        
        return successful_logos
    return []

if __name__ == "__main__":
    print("🎯 Logo Results Manager")
    print("Options:")
    print("1. Run pipeline and save results")
    print("2. Show summary of existing results")
    print("3. Access logos for similarity analysis")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        results = asyncio.run(save_results())
    elif choice == "2":
        show_summary()
    elif choice == "3":
        logos = access_logos()
        print(f"Ready for similarity analysis with {len(logos)} logos!")
    else:
        print("Invalid choice. Run again with 1, 2, or 3.")
