-- ============================================================
-- OSM Road Network + POI Pipeline — Generic City Map
-- ============================================================
--
-- Usage (psql):
--
--   psql -d <db_name> \
--     -v city=bonn \
--     -v min_lon=7.0200 -v min_lat=50.6800 \
--     -v max_lon=7.1500 -v max_lat=50.7700 \
--     -f city_map.sql
--
--   psql -d <db_name> \
--     -v city=hamburg \
--     -v min_lon=9.7300 -v min_lat=53.3900 \
--     -v max_lon=10.3300 -v max_lat=53.7400 \
--     -f city_map.sql
--
--   psql -d <db_name> \
--     -v city=bamberg \
--     -v min_lon=10.8600 -v min_lat=49.8700 \
--     -v max_lon=11.0100 -v max_lat=49.9600 \
--     -f city_map.sql
--
-- Key design: FULLY CONNECTED network for all modes.
-- No NULL time values — penalty speeds enforce mode separation:
--   Car on pedestrian/footway/cycleway : 5.0 km/h  (creep)
--   Car on steps                       : 0.001 km/h (impassable)
--   Foot/bike on motorway              : 0.001 km/h (impassable)
--   Bike on footway/pedestrian         : 5.0 km/h  (dismount)
--   Foot everywhere else               : 5.0 km/h  (walk)
--
-- Every node is reachable from every other node for every mode.
-- Dijkstra always finds a path. Time penalties make inappropriate
-- roads prohibitively expensive without disconnecting the graph.
-- ============================================================

-- ============================================================
-- STEP 0: Cleanup
-- ============================================================
DROP TABLE IF EXISTS roads_net               CASCADE;
DROP TABLE IF EXISTS roads_net_vertices_pgr  CASCADE;
DROP TABLE IF EXISTS pois                    CASCADE;
DROP TABLE IF EXISTS home_nodes              CASCADE;
DROP TABLE IF EXISTS transport_nodes         CASCADE;
DROP TABLE IF EXISTS pt_route_stops          CASCADE;
DROP TABLE IF EXISTS pt_connectivity         CASCADE;
DROP TABLE IF EXISTS nearest_road            CASCADE;
DROP TABLE IF EXISTS network_edges           CASCADE;
DROP TABLE IF EXISTS network_nodes           CASCADE;

-- ============================================================
-- STEP 0.5: Bounding Box
-- ============================================================
-- Note: psql variables are substituted as plain text.
-- Casting to float8 here ensures correct type for ST_MakeEnvelope.
-- ============================================================
CREATE TEMP TABLE city_bbox AS
SELECT ST_MakeEnvelope(
    :'min_lon'::float8,
    :'min_lat'::float8,
    :'max_lon'::float8,
    :'max_lat'::float8,
    4326
) AS geom;

SELECT
    :'city'                                             AS city,
    ROUND((ST_Area(geom::geography) / 1000000)::numeric, 2) AS area_km2
FROM city_bbox;

-- ============================================================
-- STEP 1: Road Network
-- ============================================================
CREATE TABLE roads_net AS
SELECT
    l.osm_id,
    l.way,
    l.highway,
    l.maxspeed                                          AS maxspeed_raw,
    l.oneway                                            AS oneway_raw,
    l.lanes                                             AS lanes_raw,
    l.bicycle                                           AS bicycle_raw,
    l.foot                                              AS foot_raw,
    l.tags->'bus'                                       AS bus_raw,
    l.surface                                           AS surface_raw,
    l.tags->'junction'                                  AS junction_tag,

    COALESCE(
        NULLIF(regexp_replace(l.maxspeed, '[^0-9]', '', 'g'), '')::integer,
        CASE l.highway
            WHEN 'motorway'         THEN 110
            WHEN 'motorway_link'    THEN 80
            WHEN 'primary'          THEN 50
            WHEN 'primary_link'     THEN 50
            WHEN 'secondary'        THEN 50
            WHEN 'secondary_link'   THEN 50
            WHEN 'tertiary'         THEN 50
            WHEN 'tertiary_link'    THEN 50
            WHEN 'residential'      THEN 30
            WHEN 'living_street'    THEN 10
            WHEN 'unclassified'     THEN 30
            WHEN 'cycleway'         THEN 15
            WHEN 'footway'          THEN 5
            WHEN 'path'             THEN 5
            WHEN 'pedestrian'       THEN 5
            WHEN 'steps'            THEN 3
            ELSE 30
        END
    )                                                   AS maxspeed_final,

    COALESCE(
        NULLIF(l.oneway, ''),
        CASE
            WHEN l.highway IN ('motorway','motorway_link') THEN 'yes'
            ELSE 'no'
        END
    )                                                   AS oneway_final,

    COALESCE(
        NULLIF(l.lanes, '')::integer,
        CASE l.highway
            WHEN 'motorway'         THEN 3
            WHEN 'motorway_link'    THEN 2
            WHEN 'primary'          THEN 2
            WHEN 'primary_link'     THEN 2
            WHEN 'secondary'        THEN 2
            WHEN 'secondary_link'   THEN 1
            WHEN 'tertiary'         THEN 1
            WHEN 'tertiary_link'    THEN 1
            WHEN 'residential'      THEN 1
            WHEN 'living_street'    THEN 1
            WHEN 'unclassified'     THEN 1
            ELSE 1
        END
    )                                                   AS lanes_final,

    COALESCE(
        NULLIF(l.bicycle, ''),
        CASE
            WHEN l.highway IN ('motorway','motorway_link') THEN 'no'
            WHEN l.highway = 'steps'                       THEN 'no'
            ELSE 'yes'
        END
    )                                                   AS bicycle_final,

    COALESCE(
        NULLIF(l.foot, ''),
        CASE
            WHEN l.highway IN ('motorway','motorway_link') THEN 'no'
            ELSE 'yes'
        END
    )                                                   AS foot_final,

    COALESCE(
        NULLIF(l.tags->'bus', ''),
        CASE
            WHEN l.highway IN ('motorway','motorway_link',
                               'living_street','footway','pedestrian',
                               'steps','cycleway','path')  THEN 'no'
            WHEN l.highway IN ('primary','primary_link',
                               'secondary','secondary_link',
                               'tertiary','tertiary_link')  THEN 'yes'
            ELSE 'no'
        END
    )                                                   AS bus_final,

    (l.maxspeed IS NULL OR l.maxspeed = '')             AS maxspeed_inferred,
    (l.oneway   IS NULL OR l.oneway   = '')             AS oneway_inferred,
    (l.lanes    IS NULL OR l.lanes    = '')             AS lanes_inferred,
    (l.bicycle  IS NULL OR l.bicycle  = '')             AS bicycle_inferred,
    (l.foot     IS NULL OR l.foot     = '')             AS foot_inferred,
    (l.tags->'bus' IS NULL OR l.tags->'bus' = '')       AS bus_inferred

FROM osm_line l
CROSS JOIN city_bbox b
WHERE l.highway IN (
    'motorway','motorway_link',
    'primary','primary_link',
    'secondary','secondary_link',
    'tertiary','tertiary_link',
    'residential','living_street','unclassified',
    'cycleway','footway','path','pedestrian','steps'
)
AND ST_Intersects(l.way, b.geom);

ALTER TABLE roads_net ADD COLUMN source INTEGER;
ALTER TABLE roads_net ADD COLUMN target INTEGER;
ALTER TABLE roads_net ADD COLUMN id SERIAL PRIMARY KEY;

SELECT COUNT(*) AS roads_extracted FROM roads_net;

-- ============================================================
-- STEP 2: Topology (0.00005 deg = ~5m in WGS84)
-- ============================================================
SELECT pgr_createTopology(
    'roads_net',
    0.00005,
    'way',
    'osm_id',
    'source',
    'target',
    rows_where := 'true',
    clean := true
);

SELECT COUNT(*) AS topology_vertices FROM roads_net_vertices_pgr;

-- ============================================================
-- STEP 3: POIs (point + polygon branches)
-- ============================================================
CREATE TABLE pois AS

SELECT
    p.osm_id,
    p.way                                               AS geom,
    'point'                                             AS geom_source,
    p.amenity, p.tourism, p.office, p.shop,
    p.building, p.leisure,
    NULLIF(p.tags->'opening_hours', '')                 AS opening_hours_raw,
    CASE WHEN COALESCE(p.tags->'access','') IN ('private','no')
        THEN true ELSE false END                        AS is_private,
    CASE
        WHEN p.amenity = 'restaurant'        THEN 'sustenance_restaurant'
        WHEN p.amenity = 'fast_food'         THEN 'sustenance_fastfood'
        WHEN p.amenity = 'cafe'              THEN 'sustenance_cafe'
        WHEN p.amenity = 'bar'              THEN 'sustenance_bar'
        WHEN p.amenity = 'pub'               THEN 'sustenance_pub'
        WHEN p.amenity = 'biergarten'        THEN 'sustenance_biergarten'
        WHEN p.amenity = 'food_court'        THEN 'sustenance_food_court'
        WHEN p.amenity = 'kindergarten'      THEN 'education_kindergarten'
        WHEN p.amenity = 'school'            THEN 'education_school'
        WHEN p.amenity = 'college'           THEN 'education_college'
        WHEN p.amenity = 'university'        THEN 'education_university'
        WHEN p.amenity = 'library'           THEN 'education_library'
        WHEN p.building IN ('school','college','university') THEN 'education_school'
        WHEN p.office = 'government'         THEN 'work_government'
        WHEN p.office = 'research'           THEN 'work_research'
        WHEN p.office = 'educational'        THEN 'work_education'
        WHEN p.office IS NOT NULL AND p.office != '' THEN 'work_office'
        WHEN p.building IN ('office','commercial') THEN 'work_office'
        WHEN p.building IN ('industrial','warehouse') THEN 'work_industrial'
        WHEN p.building = 'retail'           THEN 'work_retail'
        WHEN p.amenity = 'hospital'          THEN 'healthcare_hospital'
        WHEN p.amenity = 'clinic'            THEN 'healthcare_clinic'
        WHEN p.amenity = 'doctors'           THEN 'healthcare_doctor'
        WHEN p.amenity = 'dentist'           THEN 'healthcare_dentist'
        WHEN p.amenity = 'veterinary'        THEN 'healthcare_veterinary'
        WHEN p.building IN ('hospital','clinic') THEN 'healthcare_hospital'
        WHEN p.amenity = 'pharmacy'          THEN 'errand_pharmacy'
        WHEN p.amenity = 'bank'              THEN 'errand_bank'
        WHEN p.amenity = 'atm'              THEN 'errand_atm'
        WHEN p.amenity = 'post_office'       THEN 'errand_post'
        WHEN p.amenity = 'townhall'          THEN 'errand_townhall'
        WHEN p.amenity = 'courthouse'        THEN 'errand_courthouse'
        WHEN p.amenity = 'police'            THEN 'errand_police'
        WHEN p.leisure = 'park'              THEN 'leisure_park'
        WHEN p.leisure IN ('pitch','sports_centre',
             'stadium','sports_hall')        THEN 'leisure_sports'
        WHEN p.leisure = 'fitness_centre'    THEN 'leisure_fitness'
        WHEN p.leisure = 'playground'        THEN 'leisure_playground'
        WHEN p.leisure = 'garden'            THEN 'leisure_garden'
        WHEN p.leisure = 'swimming_pool'     THEN 'leisure_pool'
        WHEN p.amenity = 'park'              THEN 'leisure_park'
        WHEN p.amenity = 'cinema'            THEN 'entertainment_cinema'
        WHEN p.amenity = 'theatre'           THEN 'entertainment_theatre'
        WHEN p.amenity = 'arts_centre'       THEN 'entertainment_arts'
        WHEN p.amenity = 'nightclub'         THEN 'entertainment_nightclub'
        WHEN p.amenity = 'community_centre'  THEN 'entertainment_community'
        WHEN p.amenity = 'casino'            THEN 'entertainment_casino'
        WHEN p.amenity = 'place_of_worship'  THEN 'social_worship'
        WHEN p.building IN ('church','cathedral',
             'mosque','temple','synagogue')  THEN 'social_worship'
        WHEN p.amenity = 'social_centre'     THEN 'social_centre'
        WHEN p.tourism = 'museum'            THEN 'tourism_museum'
        WHEN p.tourism = 'gallery'           THEN 'tourism_gallery'
        WHEN p.tourism = 'attraction'        THEN 'tourism_attraction'
        WHEN p.tourism = 'viewpoint'         THEN 'tourism_viewpoint'
        WHEN p.shop = 'supermarket'          THEN 'shop_supermarket'
        WHEN p.shop = 'convenience'          THEN 'shop_convenience'
        WHEN p.shop = 'bakery'               THEN 'shop_bakery'
        WHEN p.shop = 'mall'                 THEN 'shop_mall'
        WHEN p.shop = 'department_store'     THEN 'shop_department'
        WHEN p.shop = 'clothes'              THEN 'shop_clothing'
        WHEN p.shop = 'books'                THEN 'shop_books'
        WHEN p.shop = 'hairdresser'          THEN 'shop_hairdresser'
        WHEN p.shop IS NOT NULL AND p.shop != '' THEN 'shop_other'
        WHEN p.amenity IS NOT NULL AND p.amenity != '' THEN 'amenity_other'
        WHEN p.tourism IS NOT NULL AND p.tourism != '' THEN 'tourism_other'
        WHEN p.leisure IS NOT NULL AND p.leisure != '' THEN 'leisure_other'
        ELSE 'uncategorized'
    END                                                 AS poi_category

FROM osm_point p
CROSS JOIN city_bbox b
WHERE COALESCE(p.tags->'access', '') NOT IN ('private','no')
  AND COALESCE(p.amenity, '') NOT IN (
      'bench','waste_basket','bicycle_parking','recycling',
      'parking_entrance','vending_machine','charging_station',
      'parking','post_box','telephone','clock','drinking_water',
      'waste_disposal','grit_bin','parcel_locker','loading_dock'
  )
  AND (
      (p.amenity  IS NOT NULL AND p.amenity  != '')
   OR (p.tourism  IS NOT NULL AND p.tourism  != '')
   OR (p.shop     IS NOT NULL AND p.shop     != '')
   OR (p.office   IS NOT NULL AND p.office   != '')
   OR (p.leisure  IS NOT NULL AND p.leisure  != '')
   OR (p.building IN ('office','commercial','industrial','warehouse','retail',
                      'government','hospital','clinic','school','university',
                      'college','church','cathedral','mosque','temple',
                      'synagogue','stadium','sports_hall'))
  )
  AND ST_Within(p.way, b.geom)

UNION ALL

SELECT
    p.osm_id,
    ST_Centroid(p.way)                                  AS geom,
    'polygon'                                           AS geom_source,
    p.amenity, p.tourism, p.office, p.shop,
    p.building, p.leisure,
    NULLIF(p.tags->'opening_hours', '')                 AS opening_hours_raw,
    CASE WHEN COALESCE(p.tags->'access','') IN ('private','no')
        THEN true ELSE false END                        AS is_private,
    CASE
        WHEN p.amenity = 'restaurant'        THEN 'sustenance_restaurant'
        WHEN p.amenity = 'fast_food'         THEN 'sustenance_fastfood'
        WHEN p.amenity = 'cafe'              THEN 'sustenance_cafe'
        WHEN p.amenity = 'bar'              THEN 'sustenance_bar'
        WHEN p.amenity = 'pub'               THEN 'sustenance_pub'
        WHEN p.amenity = 'biergarten'        THEN 'sustenance_biergarten'
        WHEN p.amenity = 'food_court'        THEN 'sustenance_food_court'
        WHEN p.amenity = 'kindergarten'      THEN 'education_kindergarten'
        WHEN p.amenity = 'school'            THEN 'education_school'
        WHEN p.amenity = 'college'           THEN 'education_college'
        WHEN p.amenity = 'university'        THEN 'education_university'
        WHEN p.amenity = 'library'           THEN 'education_library'
        WHEN p.building IN ('school','college','university') THEN 'education_school'
        WHEN p.office = 'government'         THEN 'work_government'
        WHEN p.office = 'research'           THEN 'work_research'
        WHEN p.office = 'educational'        THEN 'work_education'
        WHEN p.office IS NOT NULL AND p.office != '' THEN 'work_office'
        WHEN p.building IN ('office','commercial') THEN 'work_office'
        WHEN p.building IN ('industrial','warehouse') THEN 'work_industrial'
        WHEN p.building = 'retail'           THEN 'work_retail'
        WHEN p.amenity = 'hospital'          THEN 'healthcare_hospital'
        WHEN p.amenity = 'clinic'            THEN 'healthcare_clinic'
        WHEN p.amenity = 'doctors'           THEN 'healthcare_doctor'
        WHEN p.amenity = 'dentist'           THEN 'healthcare_dentist'
        WHEN p.amenity = 'veterinary'        THEN 'healthcare_veterinary'
        WHEN p.building IN ('hospital','clinic') THEN 'healthcare_hospital'
        WHEN p.amenity = 'pharmacy'          THEN 'errand_pharmacy'
        WHEN p.amenity = 'bank'              THEN 'errand_bank'
        WHEN p.amenity = 'atm'              THEN 'errand_atm'
        WHEN p.amenity = 'post_office'       THEN 'errand_post'
        WHEN p.amenity = 'townhall'          THEN 'errand_townhall'
        WHEN p.amenity = 'courthouse'        THEN 'errand_courthouse'
        WHEN p.amenity = 'police'            THEN 'errand_police'
        WHEN p.leisure = 'park'              THEN 'leisure_park'
        WHEN p.leisure IN ('pitch','sports_centre',
             'stadium','sports_hall')        THEN 'leisure_sports'
        WHEN p.leisure = 'fitness_centre'    THEN 'leisure_fitness'
        WHEN p.leisure = 'playground'        THEN 'leisure_playground'
        WHEN p.leisure = 'garden'            THEN 'leisure_garden'
        WHEN p.leisure = 'swimming_pool'     THEN 'leisure_pool'
        WHEN p.amenity = 'park'              THEN 'leisure_park'
        WHEN p.amenity = 'cinema'            THEN 'entertainment_cinema'
        WHEN p.amenity = 'theatre'           THEN 'entertainment_theatre'
        WHEN p.amenity = 'arts_centre'       THEN 'entertainment_arts'
        WHEN p.amenity = 'nightclub'         THEN 'entertainment_nightclub'
        WHEN p.amenity = 'community_centre'  THEN 'entertainment_community'
        WHEN p.amenity = 'casino'            THEN 'entertainment_casino'
        WHEN p.amenity = 'place_of_worship'  THEN 'social_worship'
        WHEN p.building IN ('church','cathedral',
             'mosque','temple','synagogue')  THEN 'social_worship'
        WHEN p.amenity = 'social_centre'     THEN 'social_centre'
        WHEN p.tourism = 'museum'            THEN 'tourism_museum'
        WHEN p.tourism = 'gallery'           THEN 'tourism_gallery'
        WHEN p.tourism = 'attraction'        THEN 'tourism_attraction'
        WHEN p.tourism = 'viewpoint'         THEN 'tourism_viewpoint'
        WHEN p.shop = 'supermarket'          THEN 'shop_supermarket'
        WHEN p.shop = 'convenience'          THEN 'shop_convenience'
        WHEN p.shop = 'bakery'               THEN 'shop_bakery'
        WHEN p.shop = 'mall'                 THEN 'shop_mall'
        WHEN p.shop = 'department_store'     THEN 'shop_department'
        WHEN p.shop = 'clothes'              THEN 'shop_clothing'
        WHEN p.shop = 'books'                THEN 'shop_books'
        WHEN p.shop = 'hairdresser'          THEN 'shop_hairdresser'
        WHEN p.shop IS NOT NULL AND p.shop != '' THEN 'shop_other'
        WHEN p.amenity IS NOT NULL AND p.amenity != '' THEN 'amenity_other'
        WHEN p.tourism IS NOT NULL AND p.tourism != '' THEN 'tourism_other'
        WHEN p.leisure IS NOT NULL AND p.leisure != '' THEN 'leisure_other'
        ELSE 'uncategorized'
    END                                                 AS poi_category

FROM osm_polygon p
CROSS JOIN city_bbox b
WHERE COALESCE(p.tags->'access', '') NOT IN ('private','no')
  AND COALESCE(p.amenity, '') NOT IN (
      'bench','waste_basket','bicycle_parking','recycling',
      'parking_entrance','vending_machine','charging_station',
      'parking','post_box','telephone','clock','drinking_water',
      'waste_disposal','grit_bin','parcel_locker','loading_dock'
  )
  AND (
      (p.amenity  IS NOT NULL AND p.amenity  != '')
   OR (p.tourism  IS NOT NULL AND p.tourism  != '')
   OR (p.shop     IS NOT NULL AND p.shop     != '')
   OR (p.office   IS NOT NULL AND p.office   != '')
   OR (p.leisure  IS NOT NULL AND p.leisure  != '')
   OR (p.building IN ('office','commercial','industrial','warehouse','retail',
                      'government','hospital','clinic','school','university',
                      'college','church','cathedral','mosque','temple',
                      'synagogue','stadium','sports_hall'))
  )
  AND ST_Within(ST_Centroid(p.way), b.geom);

SELECT COUNT(*) AS pois_extracted FROM pois;

SELECT poi_category, COUNT(*) AS count
FROM pois
GROUP BY poi_category
ORDER BY count DESC
LIMIT 30;

-- ============================================================
-- STEP 4: Home Nodes
-- ============================================================
CREATE TABLE home_nodes AS
SELECT
    p.osm_id,
    p.building                                          AS building_type,
    NULLIF(p.tags->'building:levels', '')               AS building_levels,
    ROUND(ST_Area(p.way::geography)::numeric, 2)        AS footprint_m2,
    COALESCE(
        NULLIF(p.tags->'building:levels','')::numeric, 1
    ) * ST_Area(p.way::geography)                       AS home_weight,
    ST_Centroid(p.way)                                  AS geom
FROM osm_polygon p
CROSS JOIN city_bbox b
WHERE p.building IN (
    'residential','apartments','house',
    'detached','semidetached_house','terrace',
    'dormitory','bungalow'
)
AND COALESCE(p.tags->'access', '') NOT IN ('private','no')
AND ST_Within(ST_Centroid(p.way), b.geom);

SELECT COUNT(*) AS home_nodes_extracted FROM home_nodes;

-- ============================================================
-- STEP 5: Transport Nodes
-- ============================================================
CREATE TABLE transport_nodes AS
SELECT
    p.osm_id,
    p.way                                               AS geom,
    CASE
        WHEN p.highway = 'bus_stop'                     THEN 'transport_bus'
        WHEN p.railway = 'tram_stop'                    THEN 'transport_tram'
        WHEN p.railway IN ('station','halt')            THEN 'transport_train'
        WHEN p.amenity = 'ferry_terminal'               THEN 'transport_ferry'
        WHEN p.amenity = 'bus_station'                  THEN 'transport_busstation'
        ELSE 'transport_other'
    END                                                 AS transport_type
FROM osm_point p
CROSS JOIN city_bbox b
WHERE (
    p.highway = 'bus_stop'
    OR p.railway IN ('tram_stop','station','halt')
    OR p.amenity IN ('ferry_terminal','bus_station')
)
AND ST_Within(p.way, b.geom);

SELECT transport_type, COUNT(*) AS count
FROM transport_nodes
GROUP BY transport_type
ORDER BY count DESC;

-- ============================================================
-- STEP 6: PT Route Stops + Connectivity — STUB (not used)
-- ============================================================
-- These tables are created empty so that export_*.py loads
-- without errors. PT routing is not used by the simulation —
-- mode selection (car/bike/foot) is handled by PathLibrary.
-- ============================================================
CREATE TABLE pt_route_stops (
    route_id        bigint,
    stop_id         bigint,
    route_type      text,
    sequence        integer,
    stop_name       text,
    stop_geom       geometry(Point, 4326),
    transport_type  text
);

CREATE TABLE pt_connectivity (
    origin_stop_id  bigint,
    dest_stop_id    bigint,
    route_id        bigint,
    route_type      text,
    stop_distance   integer,
    geom_distance_m double precision
);

CREATE INDEX idx_pt_conn_origin ON pt_connectivity (origin_stop_id);
CREATE INDEX idx_pt_conn_dest   ON pt_connectivity (dest_stop_id);

-- ============================================================
-- STEP 7: Snap POIs to Roads (two-pass, car-preferred)
--
-- Pass 1: nearest car/cycle/pedestrian road within 300m
-- Pass 2: fallback — any walkable road within 500m
-- ============================================================
CREATE TEMP TABLE snap_pass1 AS
SELECT DISTINCT ON (src.osm_id)
    src.osm_id      AS poi_id,
    src.poi_category,
    r.id            AS road_id,
    r.osm_id        AS road_osm_id,
    ST_ClosestPoint(r.way, src.geom) AS snapped_geom,
    ST_Distance(src.geom::geography, r.way::geography) AS distance
FROM (
    SELECT osm_id, poi_category, geom FROM pois
    UNION ALL
    SELECT osm_id, 'home_residential'::text, geom FROM home_nodes
    UNION ALL
    SELECT osm_id, transport_type, geom FROM transport_nodes
) src
CROSS JOIN LATERAL (
    SELECT id, osm_id, way
    FROM roads_net r
    WHERE r.highway IN (
        'motorway','motorway_link',
        'primary','primary_link',
        'secondary','secondary_link',
        'tertiary','tertiary_link',
        'residential','living_street','unclassified',
        'cycleway','pedestrian'
    )
    AND ST_DWithin(src.geom, r.way, 0.003)
    ORDER BY ST_Distance(src.geom, r.way)
    LIMIT 1
) r
ORDER BY src.osm_id;

SELECT COUNT(*) AS snapped_pass1 FROM snap_pass1;

CREATE TEMP TABLE snap_pass2 AS
SELECT DISTINCT ON (src.osm_id)
    src.osm_id      AS poi_id,
    src.poi_category,
    r.id            AS road_id,
    r.osm_id        AS road_osm_id,
    ST_ClosestPoint(r.way, src.geom) AS snapped_geom,
    ST_Distance(src.geom::geography, r.way::geography) AS distance
FROM (
    SELECT osm_id, poi_category, geom FROM pois
    UNION ALL
    SELECT osm_id, 'home_residential'::text, geom FROM home_nodes
    UNION ALL
    SELECT osm_id, transport_type, geom FROM transport_nodes
) src
CROSS JOIN LATERAL (
    SELECT id, osm_id, way
    FROM roads_net r
    WHERE ST_DWithin(src.geom, r.way, 0.005)
    ORDER BY ST_Distance(src.geom, r.way)
    LIMIT 1
) r
WHERE src.osm_id NOT IN (SELECT poi_id FROM snap_pass1)
ORDER BY src.osm_id;

SELECT COUNT(*) AS snapped_pass2_fallback FROM snap_pass2;

CREATE TABLE nearest_road AS
SELECT * FROM snap_pass1
UNION ALL
SELECT * FROM snap_pass2;

-- Snap audit
SELECT
    CASE
        WHEN r.highway IN (
            'motorway','motorway_link','primary','primary_link',
            'secondary','secondary_link','tertiary','tertiary_link',
            'residential','living_street','unclassified',
            'cycleway','pedestrian'
        ) THEN 'car_accessible'
        ELSE r.highway
    END AS snapped_to,
    COUNT(*) AS count
FROM nearest_road nr
JOIN roads_net r ON nr.road_id = r.id
GROUP BY snapped_to
ORDER BY count DESC;

-- ============================================================
-- STEP 8: Network Edges (unified fully-connected weights)
-- ============================================================
-- All edges have car_time_sec, bike_time_sec, foot_time_sec.
-- No NULLs — penalty speeds enforce mode separation.
-- ============================================================
CREATE TABLE network_edges AS
SELECT
    r.id                                                AS edge_id,
    r.osm_id                                            AS road_osm_id,
    r.source                                            AS node_start,
    r.target                                            AS node_end,
    r.highway,
    ST_Length(r.way::geography)                         AS edge_length_meters,
    r.maxspeed_final,
    r.oneway_final,
    r.lanes_final,
    r.bicycle_final,
    r.foot_final,
    r.bus_final,

    -- Car: normal speed on driveable roads, penalty on others
    ROUND((ST_Length(r.way::geography) / (
        CASE
            WHEN r.highway IN (
                'motorway','motorway_link',
                'primary','primary_link',
                'secondary','secondary_link',
                'tertiary','tertiary_link',
                'residential','living_street','unclassified'
            ) THEN r.maxspeed_final / 3.6
            WHEN r.highway = 'steps'
                THEN 0.001 / 3.6        -- near-impassable
            ELSE 5.0 / 3.6              -- pedestrian/cycle areas: creep at 5 km/h
        END
    ))::numeric, 2)                                     AS car_time_sec,

    -- Bike: normal speed on cycle-compatible roads, penalty on motorways/steps
    ROUND((ST_Length(r.way::geography) / (
        CASE
            WHEN r.highway IN ('motorway','motorway_link')
                THEN 0.001 / 3.6        -- near-impassable
            WHEN r.highway = 'steps'
                THEN 0.001 / 3.6        -- near-impassable
            WHEN r.bicycle_final = 'yes'
                THEN 15.0 / 3.6         -- normal cycling
            ELSE 5.0 / 3.6              -- dismount and walk
        END
    ))::numeric, 2)                                     AS bike_time_sec,

    -- Foot: normal walk speed everywhere except motorways
    ROUND((ST_Length(r.way::geography) / (
        CASE
            WHEN r.highway IN ('motorway','motorway_link')
                THEN 0.001 / 3.6        -- near-impassable
            ELSE 5.0 / 3.6              -- walk everywhere else
        END
    ))::numeric, 2)                                     AS foot_time_sec,

    EXISTS (
        SELECT 1 FROM transport_nodes t
        WHERE ST_DWithin(r.way, t.geom, 0.0005)
    )                                                   AS is_pt_served,

    (
        SELECT COUNT(*) FROM transport_nodes t
        WHERE ST_DWithin(r.way, t.geom, 0.0005)
    )                                                   AS pt_stops_nearby,

    r.maxspeed_inferred,
    r.oneway_inferred,
    r.lanes_inferred,
    r.bicycle_inferred,
    r.foot_inferred,
    r.bus_inferred

FROM roads_net r;

-- ============================================================
-- STEP 9: Network Nodes (POI -> topology vertex mapping)
-- ============================================================
-- IMPORTANT: road_id here stores the TOPOLOGY VERTEX ID
-- (roads_net.source or target), NOT the edge id (roads_net.id).
-- We pick whichever endpoint vertex is closer to the snap point.
-- ============================================================
CREATE TABLE network_nodes AS
SELECT
    nn.poi_id,
    nn.poi_category,
    CASE
        WHEN ST_Distance(nn.snapped_geom, vs.the_geom)
           <= ST_Distance(nn.snapped_geom, vt.the_geom)
        THEN r.source
        ELSE r.target
    END                                                 AS road_id,
    nn.road_osm_id,
    nn.snapped_geom,
    nn.distance                                         AS distance_to_road_meters
FROM nearest_road nn
JOIN roads_net r  ON r.id = nn.road_id
JOIN roads_net_vertices_pgr vs ON vs.id = r.source
JOIN roads_net_vertices_pgr vt ON vt.id = r.target;

-- ============================================================
-- STEP 10: Final Summary
-- ============================================================
SELECT format('=== %s FINAL SUMMARY ===', upper(:'city')) AS info;

SELECT 'Roads'                     AS table_name, COUNT(*) AS count FROM roads_net
UNION ALL
SELECT 'Topology vertices',         COUNT(*) FROM roads_net_vertices_pgr
UNION ALL
SELECT 'POIs',                      COUNT(*) FROM pois
UNION ALL
SELECT 'Home Nodes',                COUNT(*) FROM home_nodes
UNION ALL
SELECT 'Transport Nodes',           COUNT(*) FROM transport_nodes
UNION ALL
SELECT 'PT Route Stops (stub)',      COUNT(*) FROM pt_route_stops
UNION ALL
SELECT 'PT Connectivity (stub)',     COUNT(*) FROM pt_connectivity
UNION ALL
SELECT 'Network Edges',             COUNT(*) FROM network_edges
UNION ALL
SELECT 'Network Nodes (snapped)',   COUNT(*) FROM network_nodes;

SELECT '=== MODE ACCESSIBILITY ===' AS info;
SELECT
    highway,
    COUNT(*) AS edges,
    SUM(CASE WHEN car_time_sec  IS NOT NULL THEN 1 ELSE 0 END) AS car_routable,
    SUM(CASE WHEN bike_time_sec IS NOT NULL THEN 1 ELSE 0 END) AS bike_routable,
    SUM(CASE WHEN foot_time_sec IS NOT NULL THEN 1 ELSE 0 END) AS foot_routable
FROM network_edges
GROUP BY highway
ORDER BY edges DESC;

SELECT '=== POI CATEGORIES ===' AS info;
SELECT poi_category, COUNT(*) AS count
FROM pois
GROUP BY poi_category
ORDER BY count DESC;
