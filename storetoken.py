import pickle
import os

with open("tk.pkl", "wb") as f:
    pickle.dump(os.environ.get("MAPBOXTOKEN"), f)
