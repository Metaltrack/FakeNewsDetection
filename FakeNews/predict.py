import joblib
import pandas as pd

model = joblib.load(
    "fake_news_model.pkl"
)

sample = pd.DataFrame({

    "combined_text":[
        "Scientists discovered a new energy source..."
    ],

    "subject":[
        "Science"
    ],

    "year":[
        2026
    ],

    "month":[
        7
    ],

    "day":[
        3
    ]
})

prediction = model.predict(sample)[0]

if prediction == 1:
    print("True News")
else:
    print("Fake News")
