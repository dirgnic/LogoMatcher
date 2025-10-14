from logo_matcher import LogoMatcher
import json
import pandas as pd

def load_websites_from_parquet(filename: str) -> list:
    """Load website list from parquet file"""
    try:
        df = pd.read_parquet(filename)
        # Try common column names for websites
        website_columns = ['website', 'url', 'domain', 'site', 'Website', 'URL', 'Domain']
        
        for col in website_columns:
            if col in df.columns:
                websites = df[col].dropna().tolist()
                print(f"Found {len(websites)} websites in column '{col}'")
                return websites
        
        # If no standard column found, use the first column
        if len(df.columns) > 0:
            websites = df.iloc[:, 0].dropna().tolist()
            print(f"Using first column '{df.columns[0]}' with {len(websites)} websites")
            return websites
            
    except Exception as e:
        print(f"Error reading parquet file: {e}")
    
    return []

def load_websites_from_file(filename: str) -> list:
    """Load website list from text file"""
    try:
        with open(filename, 'r') as f:
            websites = [line.strip() for line in f if line.strip()]
        return websites
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

def get_sample_websites() -> list:
    """Return sample websites for testing"""
    return [
        "google.com",
        "apple.com", 
        "microsoft.com",
        "amazon.com",
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "github.com",
        "stackoverflow.com",
        "reddit.com",
        "netflix.com",
        "spotify.com",
        "adobe.com",
        "salesforce.com",
        "oracle.com"
    ]

def save_results(results: dict, filename: str = 'logo_matching_results.json'):
    """Save results to JSON file"""
    # Remove image data for JSON serialization
    clean_results = results.copy()
    clean_results['detailed_results'] = []
    
    for website_result in results['detailed_results']:
        clean_result = {k: v for k, v in website_result.items() if k != 'logo_data'}
        clean_results['detailed_results'].append(clean_result)
    
    with open(filename, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"Results saved to {filename}")

def save_groups_csv(results: dict, filename: str = 'logo_groups.csv'):
    """Save groups to CSV file"""
    import pandas as pd
    
    group_data = []
    for i, group in enumerate(results['groups'], 1):
        for website in group:
            # Find the detailed result for this website
            detailed_result = next((r for r in results['detailed_results'] if r['website'] == website), None)
            
            group_data.append({
                'group_id': i,
                'website': website,
                'group_size': len(group),
                'logo_found': detailed_result['logo_found'] if detailed_result else False,
                'logo_url': detailed_result.get('logo_url', '') if detailed_result else '',
                'error': detailed_result.get('error', '') if detailed_result else '',
                'skipped': detailed_result.get('skipped', False) if detailed_result else False,
                'attempts': detailed_result.get('attempts', 0) if detailed_result else 0
            })
    
    df = pd.DataFrame(group_data)
    df.to_csv(filename, index=False)
    print(f"Groups saved to {filename}")
    
    # Also save a summary CSV
    summary_filename = filename.replace('.csv', '_summary.csv')
    summary_data = {
        'metric': ['total_websites', 'successful_extractions', 'extraction_rate', 
                  'unparseable_domains', 'skipped_domains', 'failed_extractions', 'groups_found'],
        'value': [results['total_websites'], results['successful_extractions'], 
                 f"{results['extraction_rate']:.1f}%", results.get('unparseable_domains', 0),
                 results.get('skipped_domains', 0), results.get('failed_extractions', 0), 
                 len(results['groups'])]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary saved to {summary_filename}")

def print_results_summary(results: dict):
    """Print a summary of the results"""
    print("\n" + "="*60)
    print("LOGO MATCHING ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Total websites analyzed: {results['total_websites']}")
    print(f"Successful logo extractions: {results['successful_extractions']}")
    print(f"Extraction rate: {results['extraction_rate']:.1f}%")
    
    # Show breakdown of failures
    unparseable = results.get('unparseable_domains', 0)
    skipped = results.get('skipped_domains', 0)
    failed = results.get('failed_extractions', 0)
    
    if unparseable > 0:
        print(f"Unparseable domains: {unparseable}")
    if skipped > 0:
        print(f"Skipped domains (problematic): {skipped}")
    if failed > 0:
        print(f"Failed extractions: {failed}")
    
    print(f"Number of groups found: {len(results['groups'])}")
    
    # Show error summary if available
    if 'error_summary' in results and results['error_summary']:
        print(f"\nError Summary:")
        for error_type, count in results['error_summary'].items():
            print(f"  {error_type}: {count} websites")
    
    print(f"\nGroup Distribution:")
    group_sizes = {}
    for group in results['groups']:
        size = len(group)
        group_sizes[size] = group_sizes.get(size, 0) + 1
    
    for size in sorted(group_sizes.keys(), reverse=True):
        count = group_sizes[size]
        print(f"  Groups with {size} website(s): {count}")
    
    print(f"\nDetailed Groups:")
    multi_website_groups = [g for g in results['groups'] if len(g) > 1]
    single_website_groups = [g for g in results['groups'] if len(g) == 1]
    
    if multi_website_groups:
        print(f"\nSimilar Logo Groups ({len(multi_website_groups)} groups):")
        for i, group in enumerate(multi_website_groups, 1):
            print(f"  Group {i} ({len(group)} websites):")
            for website in group:
                print(f"    - {website}")
    
    print(f"\nUnique Logos: {len(single_website_groups)} websites")
    
    # Show unparseable domains if any
    unparseable_list = [w for w in results['detailed_results'] 
                       if w.get('error') == 'Unparseable domain format']
    if unparseable_list:
        print(f"\nUnparseable Domains ({len(unparseable_list)} domains):")
        for domain_result in unparseable_list[:5]:  # Show first 5 examples
            print(f"  - {domain_result['website']}")
        if len(unparseable_list) > 5:
            print(f"  ... and {len(unparseable_list) - 5} more")
    
    if results['extraction_rate'] < 97:
        print(f"\nFailed extractions ({results['total_websites'] - results['successful_extractions']} websites):")
        failed = [w for w in results['detailed_results'] if not w['logo_found']]
        
        # Group by error type for better reporting
        error_groups = {}
        for website_result in failed:
            error = website_result.get('error', 'Unknown error')
            error_type = error.split(':')[0] if ':' in error else error
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(website_result['website'])
        
        for error_type, websites in error_groups.items():
            print(f"  {error_type} ({len(websites)} websites):")
            for website in websites[:3]:  # Show first 3 examples
                print(f"    - {website}")
            if len(websites) > 3:
                print(f"    ... and {len(websites) - 3} more")
            print()

if __name__ == "__main__":
    print("Logo Matcher - Website Logo Similarity Analysis")
    print("=" * 50)
    
    # Initialize the logo matcher
    matcher = LogoMatcher()
    
    # Try to load websites from different sources
    websites = []
    
    # First, try the parquet file
    if not websites:
        websites = load_websites_from_parquet('logos.snappy.parquet')
    
    # Then try text file
    if not websites:
        websites = load_websites_from_file('logos_list')
    
    # Finally, use sample websites
    if not websites:
        print("No website list found. Using sample websites for demonstration.")
        websites = get_sample_websites()
    
    print(f"Starting logo matching analysis for {len(websites)} websites...")
    print("This may take several minutes depending on the number of websites.")
    
    # Run the analysis
    results = matcher.run_analysis(websites)
    
    # Print summary
    print_results_summary(results)
    
    # Save results
    save_results(results)
    save_groups_csv(results)
    
    print(f"\nAnalysis complete!")
    print(f"Check 'logo_matching_results.json' for detailed results")
    print(f"Check 'logo_groups.csv' for group information")
