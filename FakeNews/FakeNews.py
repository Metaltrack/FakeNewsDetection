from numpy import empty
import pandas as pd
import re
import os

def cleanup_stuff(text :str):
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.

#First we gotta load some data
#also gonna preprocess it here itself
def load_data(path :str, is_fake :bool = False):
    data = pd.read_csv(path)

    if not data.empty:
        print(f"Data read from CSV at path '{path}' | Data: \n")
        print(data)
    else:
        print(f"Failed to read data from CSV at path '{path}'")
        return

    #We only need data text and labels
    if is_fake:
        data["label"] = 0
    else:
        data["label"] = 1

    print("Added label to data | Data: \n")
    print(data)

    data.drop(["title", "subject", "date"])

    #data cleanup
    data["text"] = data["text"].apply(cleanup_stuff)


fake_news_data = load_data(os.path.dirname(__file__) + r"\News _dataset\Fake.csv", True)
