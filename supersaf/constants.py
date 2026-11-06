"""Physical constants and unit conversions used everywhere in the package.

Keeping them in one place means every module uses exactly the same numbers,
which matters when we compare our results with the two papers.
"""

# --- Standard atmosphere (ISA / ICAO) --------------------------------------
G0 = 9.80665            # gravitational acceleration [m s-2]
R_AIR = 287.05287       # specific gas constant of dry air [J kg-1 K-1]
GAMMA = 1.4             # ratio of specific heats of air [-]
T_SL = 288.15           # sea-level standard temperature [K]
P_SL = 101325.0         # sea-level standard pressure [Pa]

# --- Thermodynamics of moist air (used by the contrail module) -------------
CP_AIR = 1004.0         # isobaric heat capacity of air [J kg-1 K-1] (as in CoCiP)
R_VAPOUR = 461.51       # specific gas constant of water vapour [J kg-1 K-1]
EPSILON = 0.62198       # R_air / R_vapour [-] (often rounded to 0.622)

# --- Climate-metric constants (Teoh et al. 2022, Eq. 3) ---------------------
AGWP_CO2_100 = 2.92e-6  # absolute GWP of CO2, 100-yr horizon [W m-2 s kg-1]
S_EARTH = 5.101e14      # surface area of the Earth [m2]
SECONDS_PER_YEAR = 365.25 * 86400.0

# --- Combustion -------------------------------------------------------------
EI_CO2 = 3.16           # kg CO2 per kg fuel burned (tank-to-wake, as in Teoh)
MW_NO2 = 46.0055        # g/mol, NOx is reported "as NO2"
MW_N = 14.0067          # g/mol

# --- Unit conversions -------------------------------------------------------
FT = 0.3048             # 1 ft in m
LBF = 4.4482216152605   # 1 lbf in N
LB = 0.45359237         # 1 lb in kg
NM = 1852.0             # 1 nautical mile in m
LB_PER_LBF_H = LB / (LBF * 3600.0)   # 1 lb/(lbf h) expressed in kg/(N s)
