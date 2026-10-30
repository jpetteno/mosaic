# MOSAIC: Synthetic Trajectory Generation with Reinforcement Learning

The code in this repository is an implementation of our published approach:

Joshua Petteno, Rajjat Dadwal, Alishiba Dsouza, and Elena Demidova. "MOSAIC: Synthetic Trajectory Generation with Reinforcement Learning". In: *The 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '26)*, November 03–06, 2026, Riverside, CA, USA. ACM, New York, NY, USA, 4 pages. DOI: https://doi.org/10.1145/3841645.3843022

If you use the source code, please cite our research (see [Citation](#citation)).

## Notes

1) This repository does not contain any data. MOSAIC requires only freely available OpenStreetMap data: a road network and a point-of-interest (POI) set for the urban area of interest. No human mobility data is required at any stage. `map/fetch_osm_data.py` retrieves an extract for any bounding box from [Geofabrik](https://download.geofabrik.de/) and clips it to that box; see [Getting city data](#getting-city-data). The areas used in the paper are Bonn (7.020, 50.680, 7.150, 50.770 in WGS84) and Cologne (6.792, 50.879, 7.106, 50.996 in WGS84).

2) OpenStreetMap data is copyrighted OpenStreetMap contributors and available from https://www.openstreetmap.org under the Open Database License (ODbL), https://opendatacommons.org/licenses/odbl/. Any data derived from an OSM extract using this pipeline remains subject to that license. The OpenStreetMap name is a trademark of the OpenStreetMap Foundation; this work is not endorsed by or affiliated with the OpenStreetMap Foundation.

3) Building a city from scratch additionally requires PostgreSQL with the PostGIS and pgRouting extensions, and osm2pgsql. These are needed only for the `--pipeline` step. Training and generation run from the exported cache files alone.

4) It is recommended to run the python commands sequentially in a tmux session, as the pipeline and training steps may take hours.

## Introduction
MOSAIC generates synthetic human mobility trajectories for an urban area without requiring real trajectory data. It learns plausible daily activity patterns through reinforcement learning, grounded in the semantic structure of the urban area, and resolves each activity decision into a concrete destination and a road-network path.

## Implementation Overview
The approach follows a two-phase architecture. An Urban Region Representation (URR) initialization phase builds a semantically enriched spatial environment from road network and POI data. A Reinforcement Learning Loop (RLL) then learns a policy that schedules daily activities within that environment, and deploys the learned policy to generate trajectories. Intermediate artifacts, such as the enriched spatial caches, the trained models, and the generated trajectory logs, are stored in dedicated directories to ensure reproducibility and clear experiment tracking.

## Project Directory Structure

The repository is organized as follows. The `data/raw/<city>/`, `data/formatted/<city>/` and `output/<city>/` trees are all created by the commands described below.

```
.
├── run.py                              single entry point for all three stages
├── custom_settings.py                  the only file intended to be edited
├── requirements.txt
│
├── map/                                Urban Region Representation (URR)
│   ├── fetch_osm_data.py
│   ├── pipeline.py
│   ├── city_map.sql
│   ├── map.lua
│   ├── export_city.py
│   ├── filter_connected_pois.py
│   ├── precompute_air_distances.py
│   ├── precompute_poi_density.py
│   └── pre_compute_travel_times.py
│
├── src/
│   ├── config.py                       derived configuration and validation
│   ├── data_setup.py
│   ├── train.py
│   ├── sim.py                          synthetic trajectory generation
│   │
│   ├── simulation/                     Reinforcement Learning Loop (RLL)
│   │   ├── environment.py              environment, state transitions, action mask
│   │   ├── state_builder.py            state s shown to the agent
│   │   ├── agent_state.py              agent profile and daily needs sampler
│   │   ├── reward_function.py          reward R(s, a)
│   │   └── semantic_enrichment.py      activity hierarchy and venue-type properties
│   │
│   ├── spatial/
│   │   ├── region_representation.py    enriched graph, POI to road-node binding
│   │   ├── region_data.py              spatial cache loading
│   │   ├── poi_selector.py             POI selector (U_dist, U_var, U_pop)
│   │   ├── path_generator.py           mode-aware shortest paths
│   │   └── impute_favorites.py         per-agent anchor locations
│   │
│   └── analysis/
│       ├── grid_layer.py               grid abstraction
│       ├── grid_analysis.py            grid-level outputs and figures
│       └── validate_trajectories.py    evaluation metrics
│
├── data/
│   ├── raw/<city>/                     created by fetch_osm_data.py
│   │   ├── <city>.osm.pbf
│   │   └── config.json
│   └── formatted/<city>/               created by --pipeline
│       ├── spatial_cache.pkl
│       ├── spatial_cache_filtered.pkl
│       ├── air_distance_cache.pkl
│       ├── poi_density_cache.pkl
│       └── path_library.pkl
│
└── output/<city>/
    ├── model/
    │   └── model_{0|1|2|...}.zip       created by --train
    └── sim/
        └── model_{i}_sim_{j}/          created by --sim
            ├── data/
            │   ├── activity_log.csv
            │   ├── travel_log.csv
            │   ├── hourly_stamps.csv
            │   ├── paths.csv
            │   ├── poi_list.csv
            │   └── simulation_metadata.csv
            ├── grid_data/              created by --use_grid
            │   ├── data/
            │   ├── plots/
            │   └── maps/
            ├── plots/
            ├── maps/
            └── summaries/
```

## Folder Descriptions

**map/**
The Urban Region Representation phase. Imports an OSM extract into PostgreSQL, builds the routable road network and the POI tables, classifies each POI into the activity hierarchy, binds each POI to its nearest road-network node, and exports simulation-ready caches.
  + `fetch_osm_data.py` – retrieves the OSM extract for a bounding box from Geofabrik and clips it to that box, writing `data/raw/<city>/`
  + `city_map.sql` – road network, POI, home node and transport node extraction, and the venue-type classification
  + `export_city.py` – exports the PostgreSQL tables into the spatial cache
  + `filter_connected_pois.py` – restricts POIs to the largest connected road component
  + `precompute_air_distances.py`, `precompute_poi_density.py` – POI distance matrix and local density scores

**src/simulation/**
The Reinforcement Learning Loop.
  + `environment.py` – the environment: state transitions, episode structure, and the action mask that hides implausible activities
  + `state_builder.py` – assembles the state *s* (spatial, temporal and behavioral context)
  + `agent_state.py` – the agent profile, and the daily needs sampler that draws a fresh activity profile each simulated day
  + `reward_function.py` – the phase-dependent reward, the end-of-day bonus and the per-unmet-activity penalty
  + `semantic_enrichment.py` – the activity hierarchy (5 activity types, 39 venue types) and the behavioral properties attached to each venue type: opening hours, time-dependent preference weights, duration distributions, cooldowns, and popularity importance

**src/spatial/**
  + `region_representation.py` – the enriched graph: the semantically typed POI set embedded in the road network, with per-mode travel-time weights
  + `poi_selector.py` – resolves an activity decision into a concrete destination, scoring candidates on mode-aware travel time, variety and local popularity
  + `path_generator.py` – transportation-mode-aware shortest paths over the road network, with a reusable path cache

**src/analysis/**
  + `grid_layer.py` – fixed grid over the city bounding box (`CELL_SIZE_M`, 500 m in the paper)
  + `grid_analysis.py` – cell visit counts, origin–destination flows, and the corresponding figures and maps
  + `validate_trajectories.py` – trajectory property and spatial validation metrics

**data/formatted/<city>/**
The exported spatial caches. `spatial_cache_filtered.pkl` holds the POI set, road network and POI-to-road-node mapping; the remaining files are precomputed distance, density and path caches that exist purely to speed up generation and are rebuilt automatically if deleted.

**output/<city>/**
Trained models and generated trajectories. See *Output Format* below.

## Repository Usage

All python packages can be installed in a virtual environment by running the following command in the terminal:

```
pip install -r requirements.txt
```

Main dependencies include Python 3.13.5, PyTorch 2.6.0, `stable-baselines3` 2.7.0, `sb3-contrib` 2.7.1 and `gymnasium` 1.2.2. Building a city additionally requires PostgreSQL with the PostGIS and pgRouting extensions, and osm2pgsql; these are not Python packages and must be installed separately.

### Getting city data

MOSAIC needs two files per city in `data/raw/<city>/`: an OSM extract (`.pbf`) covering the area of interest, and a `config.json` giving that extract's bounding box. Both are produced for any bounding box by:

```
python map/fetch_osm_data.py --bbox 7.02,50.68,7.15,50.77 --city bonn
```

The bounding box is given as `min_lon,min_lat,max_lon,max_lat` in WGS84. The command queries the [Geofabrik](https://download.geofabrik.de/) index, downloads the smallest published region extract that fully contains the box, and clips it to exactly that box, writing:

```
data/raw/bonn/bonn.osm.pbf
data/raw/bonn/config.json          {"bbox": [7.02, 50.68, 7.15, 50.77]}
```

The pipeline below can then be run directly. Two practical notes: the downloaded region extract is cached under `data/raw/_geofabrik_cache/` and may be deleted once the clipped extract exists, and the clipping step calls `osmium`, a system package rather than a Python one (`conda install -c conda-forge osmium-tool`, or `apt install osmium-tool`).

A `.pbf` obtained from any other OpenStreetMap extract provider works equally well; in that case place it at `data/raw/<city>/<city>.osm.pbf` and write the `config.json` by hand, giving the bounding box of the extract you downloaded.

### Building the Urban Region Representation

The following command imports the extract into PostgreSQL, builds the road network and POI tables, and exports the simulation-ready caches to `data/formatted/<city>/`:

```
python run.py --pipeline bonn
```

The database password is read from `DB_PASSWORD` in `custom_settings.py`. The pipeline can be resumed from any step if interrupted:

```
python run.py --pipeline bonn --from-step 5
```

### Training the policy

Once the caches exist, the MaskablePPO policy is trained with:

```
python run.py --train bonn
python run.py --train bonn --timesteps 2000000 --n_envs 12
```

The model is saved to `output/<city>/model/model_{i}.zip`, with a new index per run.

### Generating synthetic trajectories

```
python run.py --sim bonn
python run.py --sim bonn --model 0 --n_sims 1000 --use_grid
```

Without `--model`, the latest trained model is used. `--n_sims` sets the number of agents, and `--use_grid` additionally produces the grid-level outputs. Results are written to `output/<city>/sim/model_{i}_sim_{j}/`.

### Switching cities

Each city is independent; no configuration file needs to change:

```
python run.py --pipeline cologne
python run.py --train cologne
python run.py --sim cologne
```

### Adjusting the activity hierarchy

`POI_INCLUDE` in `custom_settings.py` controls which OSM venue types enter the simulation. Commenting out an entry removes that venue type from the generated trajectories. The behavioral properties of each venue type are defined in `src/simulation/semantic_enrichment.py`.

## Output Format

The per-agent activity and travel logs are written at grid-cell resolution: the exact coordinates used internally to run the simulation are replaced by a grid cell before either log is written.

  + `activity_log.csv` – one row per activity, with the activity type, venue type, duration, and the grid cell (`cell_id`) in which the activity takes place
  + `travel_log.csv` – one row per trip, with the transportation mode, duration, distance, and the origin and destination grid cells (`origin_cell`, `dest_cell`)
  + `hourly_stamps.csv` – the activity type of each agent at each hour of the simulated period
  + `simulation_metadata.csv` – per-agent summary counts

Two further files describe the city rather than the agents, and do carry coordinates, since the road geometry is what makes the trajectories usable for a downstream spatial task:

  + `paths.csv` – one row per distinct route, with its origin and destination POI, mode, and the road-network geometry as a WKT `LINESTRING` in WGS84
  + `poi_list.csv` – the POI set of the city, with each POI's category and its longitude and latitude

POI identifiers in all of these files are internal identifiers assigned during export; they are not OpenStreetMap object identifiers. They can, however, be resolved to a location through `poi_list.csv`, so the released set does identify the buildings the synthetic agents visit — as any OSM-derived POI set does. POI names and other free-text OSM attributes are not collected by the pipeline and do not appear in the exported caches.

With `--use_grid`, `grid_data/` additionally contains cell visit counts, an origin–destination flow matrix, and the corresponding figures and maps.

## Disclaimer

  + The OpenStreetMap name is a trademark of the OpenStreetMap Foundation and is used with their permission. We are not endorsed by or affiliated with the OpenStreetMap Foundation.
  + This code is published only to support reproducibility in research. It is not intended for any type of commercial use or placement on the market.

## Licenses

  + The files `map/city_map.sql` and `map/map.lua` contain OpenStreetMap tags. The files `custom_settings.py`, `map/export_city.py`, `src/simulation/semantic_enrichment.py` and `src/spatial/region_representation.py` contain venue categories abstracted from the OpenStreetMap POI classification. OpenStreetMap data is © OpenStreetMap contributors, made available under the terms of the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) by the OpenStreetMap Foundation (OSMF), https://www.openstreetmap.org/copyright. See `LICENSE-ODbL.txt`.
  + Any database produced by running this pipeline over an OpenStreetMap extract, including the contents of `data/formatted/<city>/` and any trajectory dataset generated from it, is a Derivative Database under the ODbL and remains subject to that license.
  + The code in this repository, except for the OpenStreetMap material identified above, is licensed under the MIT license. See `LICENSE`.

## Affiliations

The source code is being published in affiliation with the Data Science and Intelligent Systems Group (DSIS), University of Bonn, Germany (https://www.uni-bonn.de/de) and the Lamarr Institute for Machine Learning and Artificial Intelligence, Bonn, Germany (https://lamarr-institute.org/).

## Acknowledgements

This work was partially funded by mFUND, Federal Ministry of Transport (BMV), Germany ('MoToRes', 01F2271C), the Deutsche Forschungsgemeinschaft (DFG) ('RUSHMORE', 547610252) and the Lamarr Institute for Machine Learning and Artificial Intelligence, Bonn, Germany (https://lamarr-institute.org/).

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{mosaic,
  author = {Petteno, Joshua and Dadwal, Rajjat and Dsouza, Alishiba and Demidova, Elena},
  title = {{MOSAIC: Synthetic Trajectory Generation with Reinforcement Learning}},
  year = {2026},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL 2026)},
  doi = {10.1145/3841645.3843022}
}
```
