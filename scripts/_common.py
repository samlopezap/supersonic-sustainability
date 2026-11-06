"""Shared setup for the step scripts: make the package importable from the repo."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROUTE_NAME = "Paris CDG - New York JFK"
ROUTE_KM = 5840.0      # great-circle distance ~5,834 km
LOAD_FACTOR = 0.8      # same seat occupancy for every aircraft
