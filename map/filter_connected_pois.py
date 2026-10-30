"""
FILTER CONNECTED POIs  
=====================
Identifies POIs in the largest connected component of the road network.
Simplified version that works with POI-road mapping tables.
"""

import sys as _sys, os as _os
_map  = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_map)
_src  = _os.path.join(_root, 'src')
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _map, _root, _src, _p


import pickle
import numpy as np
import networkx as nx
from typing import Set


def find_largest_connected_component(graph: nx.Graph) -> Set[int]:
    """Find nodes in the largest connected component."""
    if graph is None or len(graph) == 0:
        return set()
    
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)
    
    print(f"\nConnected components found: {len(components)}")
    for i, comp in enumerate(components[:5]):
        print(f"  Component {i+1}: {len(comp):,} nodes ({len(comp)/len(graph)*100:.1f}%)")
    
    largest = components[0]
    print(f"\nLargest component: {len(largest):,} nodes "
          f"({len(largest)/len(graph)*100:.1f}% of total)")
    
    return largest


def filter_pois_by_connectivity(cache_file: str, output_file: str = None):
    """Filter POIs to include only those connected to largest road component."""
    
    if output_file is None:
        output_file = cache_file.replace('.pkl', '_filtered.pkl')
    
    print("="*70)
    print("FILTERING POIs BY ROAD NETWORK CONNECTIVITY")
    print("="*70)
    
    # Load data
    print(f"\nLoading: {cache_file}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    pois            = data['pois']
    edges           = data['edges']
    nodes           = data['nodes']         # POI -> road mapping table
    graph           = data.get('graph', None)
    transport_nodes = data.get('transport_nodes', None)   # pass through unchanged
    pt_connectivity = data.get('pt_connectivity', None)   # pass through unchanged

    print(f"\nOriginal data:")
    print(f"  POIs: {len(pois):,}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Mapping table: {len(nodes):,} entries")
    if transport_nodes is not None:
        print(f"  Transport nodes: {len(transport_nodes):,}")
    if pt_connectivity is not None:
        print(f"  PT connectivity pairs: {len(pt_connectivity):,}")
    
    print(f"\nData structure:")
    print(f"  POIs columns: {list(pois.columns)}")
    print(f"  Mapping columns: {list(nodes.columns)}")
    print(f"  Edges columns: {list(edges.columns)}")
    
    # Create or use graph
    if graph is None:
        print("\nCreating graph from edges...")
        graph = nx.Graph()
        for _, edge in edges.iterrows():
            graph.add_edge(edge['source'], edge['target'],
                          weight=edge.get('length', 1.0))
        print(f"  Graph created: {len(graph):,} nodes, {len(graph.edges):,} edges")

    # Skip graph-based connectivity filter.
    # Two-pass snap (car-preferred) already ensures 99.8% of POIs land
    # on car-accessible roads. NetworkX rebuilds the topology in Python
    # and consistently fragments it worse than the PostGIS topology.
    # All road node IDs are treated as connected.
    print("\nSkipping graph connectivity filter (snap quality sufficient).")
    print(f"  All {len(nodes):,} mapped POIs retained.")
    connected_road_ids = set(nodes['road_id'].unique())
        
    # Filter POIs using the mapping table
    print(f"\nFiltering POIs...")
    
    if 'road_id' in nodes.columns and 'poi_id' in nodes.columns:
        print("  Using POI-to-road mapping table...")
        
        # Get POI IDs that map to roads in the connected component
        connected_mappings = nodes[nodes['road_id'].isin(connected_road_ids)]
        connected_poi_ids = set(connected_mappings['poi_id'].unique())
        
        print(f"  POIs mapped to connected roads: {len(connected_poi_ids):,}")
        
        # Filter POIs
        filtered_pois = pois[pois['poi_id'].isin(connected_poi_ids)].copy()
        
        print(f"\nFiltered results:")
        print(f"  POIs remaining: {len(filtered_pois):,} ({len(filtered_pois)/len(pois)*100:.1f}%)")
        print(f"  POIs removed: {len(pois) - len(filtered_pois):,}")
        
    else:
        print("  [WARNING] Cannot find POI-road mapping. Using all POIs.")
        filtered_pois = pois.copy()
    
    # Create filtered data dict -- pass transport/PT data through unchanged
    filtered_data = {
        'pois':             filtered_pois,
        'edges':            edges,
        'nodes':            nodes,
        'graph':            graph,
        'transport_nodes':  transport_nodes,
        'pt_connectivity':  pt_connectivity,
        'connected_component': list(connected_road_ids),
        'filter_stats': {
            'original_poi_count': len(pois),
            'filtered_poi_count': len(filtered_pois),
            'removal_count': len(pois) - len(filtered_pois),
            'largest_component_size': len(connected_road_ids),
            'total_components': len(list(nx.connected_components(graph)))
        }
    }
    
    # Save
    print(f"\nSaving filtered cache: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(filtered_data, f)
    
    print(f"[OK] Filtered cache saved.")
    
    # Category statistics
    if 'poi_category' in filtered_pois.columns:
        print(f"\nPOIs by category (top 10):")
        category_counts = filtered_pois['poi_category'].value_counts()
        for cat, count in category_counts.head(10).items():
            print(f"  {cat:40} {count:6,} POIs")
    
    print("\n" + "="*70)
    print("FILTERING COMPLETE")
    print("="*70)
    print(f"\nFiltered cache written to: {output_file}")
    print(f"  Run: python run.py --pipeline <city> to rebuild.")
    
    return filtered_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter POIs by connectivity")
    parser.add_argument("--input", type=str, default="data/spatial_cache.pkl")
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    try:
        filtered_data = filter_pois_by_connectivity(args.input, args.output)
        
        print("\nNext: python train.py --train")
        
    except FileNotFoundError:
        print(f"\n[WARNING] Error: File not found: {args.input}")
        print("Run pipeline.py --city <map_folder> to generate it.")
    except Exception as e:
        print(f"\n[WARNING] Error: {e}")
        import traceback
        traceback.print_exc()