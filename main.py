from ml.dataset_generator import (
    DatasetGenerator
)

generator = DatasetGenerator(
    samples=5000
)

df = generator.generate()

print(df.head())

df.to_csv(
    "missile_dataset.csv",
    index=False
)