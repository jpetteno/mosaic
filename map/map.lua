-- =============================================================
-- map.lua — osm2pgsql flex output style — city-agnostic
-- =============================================================
-- Captures:
--   osm_point   — POIs, transport stops
--   osm_line    — roads, waterways
--   osm_polygon — buildings, landuse, POI polygons
--   osm_route   — PT route relations (bus, tram, train, ferry)
--   osm_route_stop — which stops belong to which route
-- =============================================================

-- ---------------------------------------------------------------------------
-- Table definitions
-- ---------------------------------------------------------------------------

local osm_point = osm2pgsql.define_table({
    name = 'osm_point',
    ids = { type = 'node', id_column = 'osm_id' },
    columns = {
        { column = 'name',      type = 'text' },
        { column = 'amenity',   type = 'text' },
        { column = 'shop',      type = 'text' },
        { column = 'tourism',   type = 'text' },
        { column = 'leisure',   type = 'text' },
        { column = 'office',    type = 'text' },
        { column = 'building',  type = 'text' },
        { column = 'highway',   type = 'text' },
        { column = 'railway',   type = 'text' },
        { column = 'public_transport', type = 'text' },
        { column = 'tags',      type = 'hstore' },
        { column = 'way',       type = 'point', projection = 4326 },
    }
})

local osm_line = osm2pgsql.define_table({
    name = 'osm_line',
    ids = { type = 'way', id_column = 'osm_id' },
    columns = {
        { column = 'name',      type = 'text' },
        { column = 'highway',   type = 'text' },
        { column = 'waterway',  type = 'text' },
        { column = 'railway',   type = 'text' },
        { column = 'maxspeed',  type = 'text' },
        { column = 'oneway',    type = 'text' },
        { column = 'lanes',     type = 'text' },
        { column = 'bicycle',   type = 'text' },
        { column = 'foot',      type = 'text' },
        { column = 'surface',   type = 'text' },
        { column = 'tags',      type = 'hstore' },
        { column = 'way',       type = 'linestring', projection = 4326 },
    }
})

local osm_polygon = osm2pgsql.define_table({
    name = 'osm_polygon',
    ids = { type = 'area', id_column = 'osm_id' },
    columns = {
        { column = 'name',      type = 'text' },
        { column = 'amenity',   type = 'text' },
        { column = 'shop',      type = 'text' },
        { column = 'tourism',   type = 'text' },
        { column = 'leisure',   type = 'text' },
        { column = 'office',    type = 'text' },
        { column = 'building',  type = 'text' },
        { column = 'landuse',   type = 'text' },
        { column = 'admin_level', type = 'text' },
        { column = 'tags',      type = 'hstore' },
        { column = 'way',       type = 'geometry', projection = 4326 },
    }
})

-- PT route relations
local osm_route = osm2pgsql.define_table({
    name = 'osm_route',
    ids = { type = 'relation', id_column = 'osm_id' },
    columns = {
        { column = 'name',       type = 'text' },
        { column = 'route',      type = 'text' },  -- bus, tram, train, ferry
        { column = 'ref',        type = 'text' },  -- line number e.g. "61", "66"
        { column = 'operator',   type = 'text' },
        { column = 'network',    type = 'text' },
        { column = 'from_stop',  type = 'text' },
        { column = 'to_stop',    type = 'text' },
        { column = 'tags',       type = 'hstore' },
    }
})

-- Stop membership in routes (no id column — multiple rows per relation)
local osm_route_stop = osm2pgsql.define_table({
    name = 'osm_route_stop',
    ids = { type = 'relation', id_column = 'route_id' },
    columns = {
        { column = 'stop_id',    type = 'int8' },
        { column = 'stop_name',  type = 'text' },
        { column = 'route_type', type = 'text' },
        { column = 'sequence',   type = 'int4' },
    }
})


-- ---------------------------------------------------------------------------
-- Helper: check if a route type is PT
-- ---------------------------------------------------------------------------
local pt_route_types = {
    bus = true, tram = true, train = true,
    subway = true, ferry = true, light_rail = true,
    monorail = true, trolleybus = true,
}

local function is_pt_route(tags)
    return tags.route and pt_route_types[tags.route]
end


-- ---------------------------------------------------------------------------
-- Node processing
-- ---------------------------------------------------------------------------
function osm2pgsql.process_node(object)
    local t = object.tags

    -- PT stops
    local is_stop = (
        t.highway == 'bus_stop' or
        t.railway == 'tram_stop' or
        t.railway == 'station' or
        t.railway == 'halt' or
        t.public_transport == 'stop_position' or
        t.public_transport == 'platform' or
        t.amenity == 'ferry_terminal' or
        t.amenity == 'bus_station'
    )

    -- POIs
    local is_poi = (
        t.amenity or t.shop or t.tourism or
        t.leisure or t.office or
        (t.building and t.building ~= 'yes')
    )

    if is_stop or is_poi then
        osm_point:insert({
            name     = t.name,
            amenity  = t.amenity,
            shop     = t.shop,
            tourism  = t.tourism,
            leisure  = t.leisure,
            office   = t.office,
            building = t.building,
            highway  = t.highway,
            railway  = t.railway,
            public_transport = t.public_transport,
            tags     = object.tags,
            way      = object:as_point(),
        })
    end
end


-- ---------------------------------------------------------------------------
-- Way processing
-- ---------------------------------------------------------------------------
function osm2pgsql.process_way(object)
    local t = object.tags

    -- Roads and paths
    if t.highway then
        osm_line:insert({
            name     = t.name,
            highway  = t.highway,
            maxspeed = t.maxspeed,
            oneway   = t.oneway,
            lanes    = t.lanes,
            bicycle  = t.bicycle,
            foot     = t.foot,
            surface  = t.surface,
            tags     = object.tags,
            way      = object:as_linestring(),
        })
    end

    -- Waterways
    if t.waterway then
        osm_line:insert({
            name      = t.name,
            waterway  = t.waterway,
            tags      = object.tags,
            way       = object:as_linestring(),
        })
    end

    -- Railway lines
    if t.railway and (t.railway == 'tram' or t.railway == 'rail'
                      or t.railway == 'light_rail' or t.railway == 'subway') then
        osm_line:insert({
            name    = t.name,
            railway = t.railway,
            tags    = object.tags,
            way     = object:as_linestring(),
        })
    end

    -- Polygons (closed ways)
    if object.is_closed then
        local is_building = t.building and t.building ~= ''
        local is_poi = t.amenity or t.shop or t.tourism or t.leisure or t.office
        local is_landuse = t.landuse and t.landuse ~= ''
        local is_boundary = t.admin_level and t.admin_level ~= ''

        if is_building or is_poi or is_landuse or is_boundary then
            osm_polygon:insert({
                name        = t.name,
                amenity     = t.amenity,
                shop        = t.shop,
                tourism     = t.tourism,
                leisure     = t.leisure,
                office      = t.office,
                building    = t.building,
                landuse     = t.landuse,
                admin_level = t.admin_level,
                tags        = object.tags,
                way         = object:as_polygon(),
            })
        end
    end
end


-- ---------------------------------------------------------------------------
-- Relation processing
-- ---------------------------------------------------------------------------
function osm2pgsql.process_relation(object)
    local t = object.tags

    -- Multipolygon relations (buildings, landuse, admin boundaries)
    if t.type == 'multipolygon' or t.type == 'boundary' then
        local is_building = t.building and t.building ~= ''
        local is_poi = t.amenity or t.shop or t.tourism or t.leisure or t.office
        local is_landuse = t.landuse and t.landuse ~= ''
        local is_boundary = t.admin_level and t.admin_level ~= ''

        if is_building or is_poi or is_landuse or is_boundary then
            osm_polygon:insert({
                name        = t.name,
                amenity     = t.amenity,
                shop        = t.shop,
                tourism     = t.tourism,
                leisure     = t.leisure,
                office      = t.office,
                building    = t.building,
                landuse     = t.landuse,
                admin_level = t.admin_level,
                tags        = object.tags,
                way         = object:as_multipolygon(),
            })
        end
    end

    -- PT route relations
    if is_pt_route(t) then
        osm_route:insert({
            name      = t.name,
            route     = t.route,
            ref       = t.ref,
            operator  = t.operator,
            network   = t.network,
            from_stop = t['from'],
            to_stop   = t['to'],
            tags      = object.tags,
        })

        -- Write one row per stop member
        local seq = 0
        for _, member in ipairs(object.members) do
            if member.type == 'n' and (
                member.role == 'stop' or
                member.role == 'stop_entry_only' or
                member.role == 'stop_exit_only' or
                member.role == 'platform' or
                member.role == ''
            ) then
                osm_route_stop:insert({
                    stop_id    = member.ref,
                    stop_name  = nil,   -- resolved in SQL later
                    route_type = t.route,
                    sequence   = seq,
                })
                seq = seq + 1
            end
        end
    end
end