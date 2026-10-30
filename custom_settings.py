"""
CUSTOM SETTINGS
===============
This is the ONLY file you need to edit to configure and run the simulation.

Everything else -- file paths, computed constants, simulation logic -- is
handled automatically by config.py based on what you set here.

HOW TO USE
----------
1. Build your city data:
       python pipeline.py --city <your_city_folder_name>

   This downloads, clips, imports, and processes OSM data into:
       data/<your_city_folder_name>/

2. Set MAP_FOLDER below to match the folder name you used above.

3. Run training:
       python train.py --train --timesteps 1000000 --n_envs 8

4. Run simulation and analysis:
       python sim.py --model models/<your_model>.zip --n_sims 100

SECTIONS IN THIS FILE
---------------------
  1. Map selection
  2. Database connection  (pipeline only)
  3. Simulation structure
  4. Transport modes
  5. Activity durations
  6. Operating hours
  7. Daily needs
  8. Reward structure
  9. Location selection
"""

# =============================================================================
# 1. MAP SELECTION
# =============================================================================
# Set MAP_FOLDER to the name of the folder inside data/ that contains your
# city's spatial cache files. This folder is created automatically by pipeline.py
# when you run it with --city <name>.
#
# The simulation will look for all data files inside data/<MAP_FOLDER>/.
# To switch cities, change this one line -- everything else updates automatically.
#
# Examples:
#   MAP_FOLDER = "bonn"       ->  uses data/bonn/
#   MAP_FOLDER = "hamburg"    ->  uses data/hamburg/
#   MAP_FOLDER = "bamberg"    ->  uses data/bamberg/
#   MAP_FOLDER = "my_city"    ->  uses data/my_city/

MAP_FOLDER = "cologne"

# =============================================================================
# 2. DATABASE CONNECTION
# =============================================================================
# Used ONLY by pipeline.py and export_city.py when building a new city's
# spatial cache from raw OSM data. Once data/<MAP_FOLDER>/ exists with all
# cache files, these settings are not needed to run training or simulation.
#
# The database is the PostgreSQL instance where you loaded the city's OSM PBF
# file using osm2pgsql and ran city_map.sql. Each city needs its own database.

DB_HOST     = "localhost"
DB_PORT     = 5432
DB_USER     = "postgres"
DB_PASSWORD = "0000"

# =============================================================================
# 3. SIMULATION STRUCTURE
# =============================================================================

# Number of days in one training episode.
# Each episode simulates one agent's daily life for this many days.
# Longer episodes give the agent more context to learn multi-day patterns,
# but make each training update slower.
# Recommended range: 14-30 days.
EPISODE_DAYS = 30

# Fast training mode disables expensive runtime checks and trajectory tracking
# to maximise training throughput.
#   True  -- used during training (train.py sets this automatically)
#   False -- used during simulation runs where full output is needed
#            (sim.py sets this automatically; trajectories and logs are enabled)
FAST_TRAINING_MODE = True

# =============================================================================
# 4. TRANSPORT MODES
# =============================================================================

# Population prior for transport mode assignment at episode start.
# One mode is drawn per agent per episode from this distribution, then adjusted
# by commute feasibility -- e.g. a foot agent whose workplace is far away will
# be automatically upgraded to bicycle or car.
#
# Values must sum to 1.0.
# Calibrate to the modal split of your target city.
# Defaults below are based on MiD (Mobilitat in Deutschland) data for
# medium-sized German cities.
TRANSPORT_MODE_PRIOR = {
    "car":     0.40,   # 40% of agents primarily use a car
    "foot":    0.35,   # 35% walk or use public transport
    "bicycle": 0.25,   # 25% primarily cycle
}

# Effective travel speeds per mode in km/h.
# These are average speeds including stops, waiting, and access/egress time --
# not theoretical maximum speeds.
TRANSPORT_MODE_SPEEDS = {
    "car":     30.0,   # city driving with traffic, parking, and access time
    "bicycle": 15.0,   # cycling including stops and dismounting
    "foot":     5.0,   # walking pace
}

# Detour factor: corrects straight-line (air) distance to approximate road distance.
# Road routes are always longer than straight-line distance because they follow
# the street network. A factor of 1.3 means the road route is assumed to be 30%
# longer than the straight-line distance between two points.
#
# This is used for fast travel time estimation during training (avoids running
# Dijkstra on every step). Actual road-network routing via Dijkstra is used
# during simulation runs.
#
# Typical values:
#   1.2 -- dense, regular grid city (e.g. Manhattan)
#   1.3 -- typical European city (default)
#   1.4 -- sprawling or hilly city with indirect road layout
DETOUR_FACTOR = 1.3

# =============================================================================
# 5. ACTIVITY DURATIONS (hours)
# =============================================================================
# Each activity's duration is sampled from a normal distribution at the moment
# the agent starts the activity. The sampled value is clipped to [min, max].
#
# Format per activity and day type:
#   mean -- average duration in hours
#   std  -- standard deviation (controls how variable durations are day-to-day)
#   min  -- minimum possible duration after clipping
#   max  -- maximum possible duration after clipping
#   None -- activity not available on this day type (masked in action space)
#
# HOME note: HOME has no fixed duration. Agents stay home indefinitely until
# a need arises or the slot budget is consumed. HOME does not consume a slot
# and agents can leave at any time operating hours allow. No duration is
# sampled for HOME -- the None entries below exist only for schema completeness.

ACTIVITY_DURATIONS = {
    "WORK": {
        "weekday": {"mean": 8.0, "std": 0.5, "min": 4.0,  "max": 12.0},
        "weekend": None,   # work is masked on weekends
    },
    "FOOD": {
        # Covers any eating-outside event: breakfast, lunch, dinner, or cafe visit.
        # The specific sub-activity (restaurant, cafe, bar, etc.) is sampled
        # separately by semantic_enrichment.py based on time of day.
        "weekday": {"mean": 0.75, "std": 0.25, "min": 0.4,  "max": 1.75},
        "weekend": {"mean": 1.0,  "std": 0.3,  "min": 0.5,  "max": 2.0},
    },
    "ERRAND": {
        # Covers shopping, pharmacy, bank, post office, and similar tasks.
        "weekday": {"mean": 1.0, "std": 0.5,  "min": 0.5, "max": 2.0},
        "weekend": {"mean": 1.5, "std": 0.75, "min": 0.5, "max": 3.0},
    },
    "LEISURE": {
        # Covers parks, gyms, cultural venues, bars, cinemas, etc.
        "weekday": {"mean": 2.0, "std": 0.5, "min": 1.0, "max": 4.0},
        "weekend": {"mean": 4.0, "std": 1.0, "min": 2.0, "max": 8.0},
    },
    "HOME": {
        # No duration sampling for HOME -- see note above.
        "weekday": None,
        "weekend": None,
    },
}

# =============================================================================
# 6. OPERATING HOURS
# =============================================================================
# Defines the broad time windows during which each macro activity can be chosen.
# Format: (open_hour, close_hour) in 24-hour time. (0, 0) = closed all day.
#
# IMPORTANT -- how masking works for each macro:
#
#   HOME and WORK use these hours directly. HOME is always open (0-24).
#   WORK is available weekday 07:00-19:00 and closed on weekends (0, 0).
#
#   FOOD, ERRAND, and LEISURE use these hours as a FALLBACK ONLY.
#   For those three macros, the masking system checks sub-activity availability
#   from semantic_enrichment.py -- a macro is considered open if ANY of its
#   sub-activities is currently open (e.g. FOOD is open if cafes, restaurants,
#   or bars are open right now). The coarse hours below are only consulted if
#   semantic_enrichment.py has no sub-activity data for that macro, which should
#   not happen in normal operation. They are kept here for safety and
#   transparency about the intended availability windows.

OPERATING_HOURS = {
    "HOME":    {"weekday": (0,  24), "weekend": (0,  24)},  # always open
    "WORK":    {"weekday": (7,  19), "weekend": (0,   0)},  # weekdays only; (0,0) = closed
    "FOOD":    {"weekday": (6,  23), "weekend": (6,  24)},  # fallback only; sub-activities govern
    "ERRAND":  {"weekday": (8,  20), "weekend": (9,  18)},  # fallback only; sub-activities govern
    "LEISURE": {"weekday": (8,  24), "weekend": (8,  24)},  # fallback only; sub-activities govern
}

# =============================================================================
# 7. DAILY NEEDS
# =============================================================================
# Each day, the agent draws a set of needs it must satisfy before day's end.
# Needs are sampled in two independent Bernoulli stages:
#
#   Stage 1 -- Is this need active today?
#     Each need is independently drawn with probability from NEEDS_PROBABILITIES.
#     Example: on a weekday, there is a 40% chance the agent needs to eat out.
#
#   Stage 2 -- Is this need needed once or twice today?
#     If the need fired in stage 1, a second Bernoulli draw (using
#     DOUBLE_NEED_PROBS) decides whether the agent needs to satisfy it
#     once or twice. This models days with two eating-out events (breakfast
#     + dinner), two errands, etc.
#     Example: given eating out is needed, there is a 10% chance on weekdays
#     that the agent needs two separate food events.
#
# WORK is not drawn by Bernoulli -- it is always required exactly once on
# weekdays and never on weekends.
#
# The agent's daily slot budget = sum of all required counts + 1 discretionary.
# The +1 allows one extra out-of-home activity beyond strict needs.

# Stage 1: P(need is active today)
NEEDS_PROBABILITIES = {
    "weekday": {
        "eatoutside": 0.40,   # ~2 eating-out events per week
        "runerrand":  0.55,   # ~2.75 errand days per week
        "leisure":    0.50,   # ~2.5 leisure days per week
    },
    "weekend": {
        "eatoutside": 0.60,   # higher probability of eating out on weekends
        "runerrand":  0.60,   # weekend shopping and errands
        "leisure":    0.80,   # high leisure probability on weekends
    },
}

# Stage 2: P(need required twice | need is active today)
# Only applies when stage 1 fired. Keep values modest -- most days an agent
# satisfies each need exactly once.
DOUBLE_NEED_PROBS = {
    "eatoutside": {"weekday": 0.10, "weekend": 0.30},  # e.g. lunch + dinner out
    "runerrand":  {"weekday": 0.05, "weekend": 0.15},  # e.g. two separate errands
    "leisure":    {"weekday": 0.10, "weekend": 0.40},  # e.g. gym morning + bar evening
}

# =============================================================================
# 8. REWARD STRUCTURE
# =============================================================================

# Bonus awarded at end of day when ALL required needs are satisfied.
# This is the primary signal for daily goal completion. If not all needs are
# met, a penalty of -1.0 per unmet need is applied instead (see reward_function.py).
# Higher ALL_NEEDS_BONUS = stronger pressure to complete all needs every day.
ALL_NEEDS_BONUS = 3.5

# Phase-activity reward table.
# Controls WHEN activities are rewarded, independently of need satisfaction.
#
# The day is split into 6 temporal phases. Each (phase, macro) pair has a
# reward value that shapes timing -- e.g. going to WORK during morning_commute
# earns +1.0, while LEISURE before 9am earns -1.0.
#
# This reward is additive with the need-satisfaction signal. An activity that
# both satisfies a need AND happens at the right time earns both.
#
# Phases:
#   0 = morning_commute   weekday  06:00-09:00
#   1 = work_hours        weekday  09:00-17:00
#   2 = evening           weekday  17:00-22:00
#   3 = night             any day  22:00-06:00 + weekend sleep-in 06:00-10:00
#   4 = weekend_day       weekend  10:00-18:00
#   5 = weekend_evening   weekend  18:00-22:00
#
# Scale: +1.0 ideal, +0.5 acceptable, 0.0 neutral, -0.5 mildly wrong, -1.0 strongly wrong
# HOME always 0 in this table; a separate +0.5 sleep reward applies at night (21:00-06:00).

PHASE_ACTIVITY_REWARD = {
    # Phase 0: morning commute (06:00-09:00 weekday)
    0: {
        "WORK":    1.0,   # ideal -- commuting to work in the morning
        "FOOD":    0.5,   # acceptable -- breakfast before work
        "ERRAND": -1.0,   # strongly discouraged -- pre-work errands disrupt schedule
        "LEISURE":-1.0,   # strongly discouraged -- leisure before work
        "HOME":    0.0,
    },
    # Phase 1: work hours (09:00-17:00 weekday)
    1: {
        "WORK":    1.0,   # ideal -- at work during core hours
        "FOOD":    0.5,   # acceptable -- lunch break
        "ERRAND":  0.5,   # acceptable -- lunchtime errand
        "LEISURE":-0.5,   # mildly discouraged -- skipping work for leisure
        "HOME":   -0.5,   # mildly discouraged -- returning home during work hours
    },
    # Phase 2: evening (17:00-22:00 weekday)
    2: {
        "WORK":   -1.0,   # strongly discouraged -- working very late
        "FOOD":    1.0,   # ideal -- dinner prime time
        "ERRAND":  0.5,   # acceptable -- after-work errand
        "LEISURE": 1.0,   # ideal -- evening leisure prime time
        "HOME":    0.5,   # good -- winding down for the evening
    },
    # Phase 3: night (22:00-06:00 any day + weekend sleep-in 06:00-10:00)
    3: {
        "WORK":   -1.0,   # strongly discouraged -- working through the night
        "FOOD":   -0.5,   # mildly discouraged -- late-night eating unusual
        "ERRAND": -1.0,   # strongly discouraged -- nothing open at night
        "LEISURE": 0.0,   # neutral -- nightlife is legitimate
        "HOME":    0.5,   # good -- sleeping at home
    },
    # Phase 4: weekend day (10:00-18:00 weekend)
    4: {
        "WORK":    0.0,   # neutral -- WORK is masked on weekends anyway
        "FOOD":    1.0,   # ideal -- brunch / lunch
        "ERRAND":  1.0,   # ideal -- core window for weekend errands
        "LEISURE": 1.0,   # ideal -- weekend day leisure
        "HOME":    0.5,   # good -- resting is fine
    },
    # Phase 5: weekend evening (18:00-22:00 weekend)
    5: {
        "WORK":    0.0,   # neutral -- WORK is masked on weekends anyway
        "FOOD":    1.0,   # ideal -- dinner out
        "ERRAND": -0.5,   # mildly discouraged -- most shops closing after 18-19h
        "LEISURE": 1.0,   # ideal -- evening out on the weekend
        "HOME":    0.0,   # neutral
    },
}

# Minimum gap before the same macro activity can be repeated (hours).
# Prevents unrealistic back-to-back repetition of the same activity type.
# FOOD cooldown of 6h = after eating out, agent cannot eat out again for 6h.
MACRO_COOLDOWN_HOURS = {
    "FOOD":    6.0,   # realistic minimum gap between two eating-out events
    "ERRAND":  4.0,   # realistic minimum gap between two errand trips
    "LEISURE": 2.0,   # realistic minimum gap between two leisure activities
}

# =============================================================================
# 9. LOCATION SELECTION
# =============================================================================
# Controls how the location selector picks a specific POI once the RL agent
# has chosen a macro activity (e.g. the agent chose FOOD -- which restaurant?).

# Probability that the agent revisits a previously visited POI rather than
# exploring a new one. Higher = more habitual, lower = more exploratory.
# Literature: Pappalardo et al. (2015) find ~70% return rate empirically;
# this is set lower to allow spatial diversity in the synthetic population.
REVISIT_PROBABILITY = 0.30

# Number of candidate POIs scored and considered at each selection step.
# Higher values allow more diverse selection but require more scoring operations.
TOP_K_VENUES = 5

# Softmax temperature for final venue selection from the scored candidate list.
# Low  (e.g. 0.1) -- near-deterministic, always picks the highest-scored venue.
# High (e.g. 2.0) -- near-random, all candidates roughly equally likely.
VENUE_SELECTION_TEMPERATURE = 0.5

# Weights for the multi-factor venue scoring function.
# Each dimension contributes proportionally to its weight.
#   distance_from_current -- penalises venues far from the agent's current position
#   distance_from_home    -- mild gravitational pull toward the home neighbourhood
#   distance_from_work    -- mild gravitational pull toward the work neighbourhood
#   popularity            -- preference for venues in areas with higher POI density
VENUE_SCORE_WEIGHTS = {
    "distance_from_current": 5.0,   # dominant -- agents strongly prefer nearby venues
    "distance_from_home":    1.0,
    "distance_from_work":    1.0,
    "popularity":            0.5,
}

# Distance decay half-life in km per reference point.
# At half-life distance the distance score is halved; beyond that it falls further.
# Larger values = agents are willing to travel further for activities.
# Calibrate to your city's scale:
#   Bonn    (~13km wide): 20-30km half-life is appropriate
#   Hamburg (~60km wide): consider scaling up to 40-60km
DISTANCE_DECAY_HALF_LIFE = {
    "current": 20.0,   # decay from current location
    "home":    30.0,   # decay from home
    "work":    30.0,   # decay from work
}

# =============================================================================
# 10. POI CATEGORIES
# =============================================================================
# Controls which POI types are included in the simulation.
# Comment out any category you do not want agents to visit.
# Categories match the poi_category field produced by city_map.sql.
#
# To find all available categories in your city after running --pipeline:
#   python -c "import pickle; d=pickle.load(open('data/formatted/bonn/spatial_cache_filtered.pkl','rb')); print(d['pois']['poi_category'].value_counts().to_string())"

POI_INCLUDE = [
    # --- Food & drink ---
    "sustenance_restaurant",
    "sustenance_fastfood",
    "sustenance_cafe",
    "sustenance_pub",
    "sustenance_bar",
    "sustenance_biergarten",
    "sustenance_food_court",

    # --- Work & office ---
    "work_office",
    "work_government",
    "work_research",
    "work_education",
    "work_industrial",
    "work_retail",

    # --- Education ---
    "education_school",
    "education_university",
    "education_college",
    "education_library",
    "education_kindergarten",

    # --- Healthcare ---
    "healthcare_hospital",
    "healthcare_clinic",
    "healthcare_doctor",
    "healthcare_dentist",

    # --- Errands & services ---
    "errand_pharmacy",
    "errand_bank",
    "errand_post",
    "errand_townhall",
    "errand_atm",
    "errand_police",

    # --- Shopping ---
    "shop_supermarket",
    "shop_convenience",
    "shop_bakery",
    "shop_mall",
    "shop_department",
    "shop_clothing",
    "shop_books",
    "shop_hairdresser",

    # --- Leisure & sport ---
    "leisure_park",
    "leisure_garden",
    "leisure_fitness",
    "leisure_sports",
    "leisure_pool",
    "leisure_playground",

    # --- Entertainment ---
    "entertainment_cinema",
    "entertainment_theatre",
    "entertainment_arts",
    "entertainment_community",
    "entertainment_nightclub",

    # --- Social & religion ---
    "social_worship",
    "social_centre",

    # --- Tourism ---
    "tourism_museum",
    "tourism_gallery",
    "tourism_attraction",
    "tourism_viewpoint",

    # --- Home (always included -- do not remove) ---
    "home_residential",
]
