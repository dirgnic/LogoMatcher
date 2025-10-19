#!/usr/bin/env python3

import pickle
import json

def debug_logo_structure():
    """Debug the structure of our enhanced logo data"""
    
    # Load enhanced results
    with open('comprehensive_logo_extraction_fast_results.pkl', 'rb') as f:
        enhanced_data = pickle.load(f)
    
    print(" ENHANCED LOGO DATA STRUCTURE")
    print("=" * 50)
    
    successful_logos = enhanced_data.get('successful_logos', [])
    print(f"Total successful logos: {len(successful_logos)}")
    
    if successful_logos:
        # Show structure of first logo
        first_logo = successful_logos[0]
        print(f"\nFirst logo keys: {list(first_logo.keys())}")
        print(f"First logo content:")
        for key, value in first_logo.items():
            if key == 'logo_data':
                print(f"  {key}: <binary data> ({len(value)} bytes)")
            else:
                print(f"  {key}: {value}")
    
    return successful_logos

if __name__ == "__main__":
    debug_logo_structure()
