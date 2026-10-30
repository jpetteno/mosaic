"""
SPATIAL ENVIRONMENT FOR OSM DATA
=================================
Loads and manages OSM spatial data from pre-exported cache.

This module loads POIs, road networks, and routing data from
the spatial_cache.pkl file created by export_city.py.
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p


import pickle
import pandas as pd
from typing import Dict, Optional
import os


# Global cache to avoid redundant loading in parallel environments
_SPATIAL_ENV_CACHE: Dict[str, 'RegionData'] = {}


class RegionData:
    """Container for OSM spatial data"""
    
    def __init__(self, cache_file: str):
        """
        Load spatial data from cache file.
        
        Args:
            cache_file: Path to spatial_cache.pkl
        """
        self.cache_file = cache_file
        
        # Load data
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract components
        self.pois   = data['pois']              # DataFrame: all POIs + home_residential
        self.edges  = data['edges']             # DataFrame: road edges with mode weights
        self.nodes  = data['nodes']             # DataFrame: POI -> road node mapping
        self.graph  = data.get('graph', None)   # NetworkX graph

        # v4: transport and PT data
        self.transport_nodes  = data.get('transport_nodes', None)   # PT stops
        self.pt_connectivity  = data.get('pt_connectivity', None)   # stop-to-stop pairs

        # Check if this is a filtered cache
        self.is_filtered = 'connected_component' in data
        self.connected_component = data.get('connected_component', None)
        self.filter_stats = data.get('filter_stats', None)

        self.pois_by_category = self._group_pois_by_category()

        print(f"Loading spatial data from cache: {cache_file}")
        print(f"Spatial environment ready: {len(self.pois):,} POIs, {len(self.edges):,} edges")
        if self.transport_nodes is not None:
            print(f"  Transport nodes: {len(self.transport_nodes):,}")
        if self.pt_connectivity is not None:
            print(f"  PT connectivity pairs: {len(self.pt_connectivity):,}")

        if self.is_filtered:
            print(f"[OK] Using filtered cache (connectivity-checked)")
            if self.filter_stats:
                print(f"  Original POIs: {self.filter_stats['original_poi_count']:,}")
                print(f"  Filtered POIs: {self.filter_stats['filtered_poi_count']:,}")
                print(f"  Connected component: {self.filter_stats['largest_component_size']:,} nodes")
    
    def _group_pois_by_category(self) -> Dict[str, pd.DataFrame]:
        """Group POIs by their category for fast lookup - OPTIMIZED"""
        grouped = {}
        
        # Check for category column (could be 'category' or 'poi_category')
        cat_col = None
        for possible_col in ['poi_category', 'category', 'type']:
            if possible_col in self.pois.columns:
                cat_col = possible_col
                break
        
        if cat_col is not None:
            # OPTIMIZATION: Use groupby() for faster categorization
            for category, group_df in self.pois.groupby(cat_col):
                grouped[category] = group_df
        
        return grouped
    
    def get_poi_info(self, poi_id: int) -> Dict:
        """
        Get information about a specific POI.
        
        Args:
            poi_id: POI ID
            
        Returns:
            Dict with POI information
        """
        poi_row = self.pois[self.pois['poi_id'] == poi_id]
        
        if len(poi_row) == 0:
            return None
        
        return poi_row.iloc[0].to_dict()
    
    def get_pois_in_category(self, category: str) -> pd.DataFrame:
        """
        Get all POIs in a specific category.
        
        Args:
            category: Category name (e.g., 'sustenance_restaurant')
            
        Returns:
            DataFrame with POIs in this category
        """
        return self.pois_by_category.get(category, pd.DataFrame())
    
    def get_categories(self) -> list:
        """Get list of all available POI categories"""
        return list(self.pois_by_category.keys())
    
    def get_statistics(self) -> Dict:
        """Get statistics about the spatial environment"""
        stats = {
            'total_pois':            len(self.pois),
            'total_edges':           len(self.edges),
            'total_nodes':           len(self.nodes),
            'categories':            len(self.pois_by_category),
            'has_graph':             self.graph is not None,
            'transport_nodes':       len(self.transport_nodes) if self.transport_nodes is not None else 0,
            'pt_connectivity_pairs': len(self.pt_connectivity) if self.pt_connectivity is not None else 0,
        }
        
        # Category counts
        category_counts = {}
        for cat, pois_df in self.pois_by_category.items():
            category_counts[cat] = len(pois_df)
        stats['category_counts'] = category_counts
        
        return stats


def get_spatial_env(cache_file: str = None) -> RegionData:
    """
    Factory function to load spatial environment with caching.
    
    This function caches loaded environments globally to avoid redundant
    loading in parallel processes. Each subprocess will only load once.
    
    Args:
        cache_file: Path to spatial cache file
        
    Returns:
        RegionData object (cached if previously loaded)
    """
    if cache_file is None:
        from config import SPATIAL_CACHE_FILE
        cache_file = SPATIAL_CACHE_FILE
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Spatial cache file not found: {cache_file}\n"
            f"Run pipeline.py --city <map_folder> to generate it."
        )
    
    # Return cached instance if available
    if cache_file in _SPATIAL_ENV_CACHE:
        print(f"[CACHED] Using cached spatial data (already loaded in this process)")
        return _SPATIAL_ENV_CACHE[cache_file]
    
    # Load new instance and cache it
    env = RegionData(cache_file)
    _SPATIAL_ENV_CACHE[cache_file] = env
    return env


if __name__ == "__main__":
    print("="*60)
    print("TESTING RegionData")
    print("="*60)
    
    try:
        spatial_env = get_spatial_env()
        
        print("\nStatistics:")
        stats = spatial_env.get_statistics()
        print(f"  Total POIs: {stats['total_pois']:,}")
        print(f"  Total edges: {stats['total_edges']:,}")
        print(f"  Total nodes: {stats['total_nodes']:,}")
        print(f"  Categories: {stats['categories']}")
        print(f"  Has graph: {stats['has_graph']}")
        
        print("\nTop 10 categories by POI count:")
        sorted_cats = sorted(stats['category_counts'].items(), 
                           key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats[:10]:
            print(f"  {cat:30} {count:6,} POIs")
        
        print("\n[OK] Spatial environment loaded successfully.")

    except FileNotFoundError as e:
        print(f"\n[WARNING] {e}")
        print("\nTo create the spatial cache:")
        print("  1. Run: python run.py --pipeline <city>")
        print("  2. Run: python pipeline.py --city <map_folder>")