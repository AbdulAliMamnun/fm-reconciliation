"""Load TourismSmall and TourismLarge and print their basic structure.

Run from the repo root: python notebooks/00_load.py
"""
from datasetsforecast.hierarchical import HierarchicalData

for name in ["TourismSmall", "TourismLarge"]:
    Y_df, S_df, tags = HierarchicalData.load("./data", name)

    print(f"===== {name} =====")
    print(f"Y_df shape: {Y_df.shape}")
    print(f"S_df shape: {S_df.shape}")
    print(f"tag keys:   {list(tags.keys())}")
    for key, ids in tags.items():
        print(f"  {key}: {len(ids)} series")
    print("Y_df.head():")
    print(Y_df.head())
    print()
