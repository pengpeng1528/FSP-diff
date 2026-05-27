import os
 
version_file = os.path.join(os.path.dirname(__file__), "..", "VERSION")
with open(version_file, "r") as f:
    __version__ = f.read().strip() 