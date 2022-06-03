import pandas as pd
import random

if __name__ == "__main__":
    csv = pd.read_csv("datav1.csv")
    demo_labels = [random.randint(0, 1) for i in range(len(csv))]
    csv['isdefault'] = pd.DataFrame(demo_labels)
    csv.rename(columns={"text": "sen"}, inplace=True)
    csv.to_csv("defaults.csv", index=False)
    print(csv)
    print(demo_labels)