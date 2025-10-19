#!/usr/bin/env python3
"""
Test script to verify the logo matcher setup and dependencies.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    required_packages = [
        'requests',
        'bs4',
        'cv2',
        'numpy',
        'PIL',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f" {package}")
        except ImportError:
            print(f" {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def test_logo_matcher():
    """Test if the LogoMatcher class can be imported"""
    try:
        from logo_matcher import LogoMatcher
        matcher = LogoMatcher()
        print(" LogoMatcher class imported successfully")
        return True
    except Exception as e:
        print(f" Failed to import LogoMatcher: {e}")
        return False

def run_mini_test():
    """Run a mini test with a single website"""
    try:
        from logo_matcher import LogoMatcher
        
        matcher = LogoMatcher()
        print("\nRunning mini test with google.com...")
        
        result = matcher.process_website("google.com")
        
        if result['logo_found']:
            print(f" Successfully extracted logo from {result['website']}")
            print(f"  Logo URL: {result['logo_url']}")
        else:
            print(f" Failed to extract logo: {result.get('error', 'Unknown error')}")
            
        return result['logo_found']
        
    except Exception as e:
        print(f" Mini test failed: {e}")
        return False

if __name__ == "__main__":
    print("Logo Matcher Setup Test")
    print("=" * 30)
    
    print("\n1. Testing package imports...")
    missing = test_imports()
    
    if missing:
        print(f"\n Missing packages: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n2. Testing LogoMatcher import...")
    if not test_logo_matcher():
        sys.exit(1)
    
    print("\n3. Running connectivity test...")
    if run_mini_test():
        print("\n All tests passed! Your setup is ready.")
    else:
        print("\n  Setup is functional but connectivity test failed.")
        print("This might be due to network issues or website blocking.")
        
    print("\nYou can now run: python main.py")
