
import os
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. LOAD DATASET FROM THE PROJECT FOLDER
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "car.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        "car.csv was not found. Place your CSV file "
        "in the same folder as app.py."
    )

df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# --------------------------------------------------
# 2. CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

numeric_columns = ["year", "selling_price", "km_driven"]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

df = df.dropna(
    subset=["year", "selling_price", "km_driven"]
)

# Remove impossible or invalid values
df = df[
    (df["selling_price"] > 0)
    & (df["km_driven"] >= 0)
    & (df["year"] >= 1900)
    & (df["year"] <= 2100)
].copy()

categorical_columns = [
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

for column in categorical_columns:
    df[column] = (
        df[column]
        .replace(r"^\s*$", np.nan, regex=True)
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

if len(df) < 10:
    raise ValueError(
        "The dataset has too few valid rows to train a model."
    )

# --------------------------------------------------
# 4. SELECT FEATURES AND TARGET
# --------------------------------------------------

features = [
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

X = df[features]
y = df["selling_price"]

# --------------------------------------------------
# 5. SPLIT THE DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 6. PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            ["year", "km_driven"]
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# --------------------------------------------------
# 7. CREATE AND TRAIN MODEL
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training model...")
model.fit(X_train, y_train)
print("Model training completed!")

# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print(f"MAE: Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# --------------------------------------------------
# 9. PREPARE DROPDOWN OPTIONS
# --------------------------------------------------

fuel_values = sorted(df["fuel"].unique().tolist())
seller_values = sorted(df["seller_type"].unique().tolist())
transmission_values = sorted(
    df["transmission"].unique().tolist()
)
owner_values = sorted(df["owner"].unique().tolist())

# --------------------------------------------------
# 10. PREDICTION FUNCTION
# --------------------------------------------------

def predict_price(
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner
):
    try:
        if year is None or km_driven is None:
            return "Please enter the year and kilometres driven."

        if not str(year).replace(".", "", 1).isdigit():
            return "Please enter a valid manufacturing year."

        if not str(km_driven).replace(".", "", 1).isdigit():
            return "Please enter valid kilometres driven."

        year = int(float(year))
        km_driven = float(km_driven)

        if year < 1900 or year > 2100:
            return "Please enter a valid manufacturing year."

        if km_driven < 0:
            return "Kilometres driven cannot be negative."

        current_year = pd.Timestamp.now().year

        if year > current_year:
            return "Manufacturing year cannot be in the future."

        input_data = pd.DataFrame({
            "year": [year],
            "km_driven": [km_driven],
            "fuel": [str(fuel)],
            "seller_type": [str(seller_type)],
            "transmission": [str(transmission)],
            "owner": [str(owner)]
        })

        predicted_price = float(
            model.predict(input_data)[0]
        )

        predicted_price = max(0, predicted_price)

        return (
            "Estimated Car Resale Price\n\n"
            f"Rs. {predicted_price:,.0f}\n"
            f"Approximately Rs. "
            f"{predicted_price / 100000:.2f} lakhs\n\n"
            "This is an estimate, not a guaranteed selling price."
        )

    except Exception as error:
        return f"Prediction error: {error}"

# --------------------------------------------------
# 11. CREATE INTERACTIVE GRADIO WEBSITE
# --------------------------------------------------

with gr.Blocks(
    title="Car Resale Price Predictor",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown(
        """
        # Car Resale Price Prediction

        ### Machine Learning Based Price Estimator

        Enter your vehicle details below to estimate
        its resale price.
        """
    )

    gr.Markdown(
        f"""
        **Algorithm:** Random Forest Regression

        **Training records:** {len(X_train)}

        **Testing records:** {len(X_test)}

        **R² Score:** {r2:.4f}
        """
    )

    with gr.Row():
        with gr.Column():
            year_input = gr.Number(
                label="Manufacturing Year",
                value=2018,
                precision=0
            )

            km_input = gr.Number(
                label="Kilometres Driven",
                value=50000,
                minimum=0,
                precision=0
            )

            fuel_input = gr.Dropdown(
                choices=fuel_values,
                label="Fuel Type",
                value=fuel_values[0]
            )

        with gr.Column():
            seller_input = gr.Dropdown(
                choices=seller_values,
                label="Seller Type",
                value=seller_values[0]
            )

            transmission_input = gr.Dropdown(
                choices=transmission_values,
                label="Transmission",
                value=transmission_values[0]
            )

            owner_input = gr.Dropdown(
                choices=owner_values,
                label="Previous Owner",
                value=owner_values[0]
            )

    predict_button = gr.Button(
        "Predict Resale Price",
        variant="primary"
    )

    output = gr.Textbox(
        label="Prediction Result",
        lines=4
    )

    predict_button.click(
        fn=predict_price,
        inputs=[
            year_input,
            km_input,
            fuel_input,
            seller_input,
            transmission_input,
            owner_input
        ],
        outputs=output
    )

    gr.Markdown(
        f"""
        ---

        ### Model Evaluation

        - **Mean Absolute Error (MAE):** Rs. {mae:,.2f}
        - **Root Mean Squared Error (RMSE):** Rs. {rmse:,.2f}
        - **R² Score:** {r2:.4f}

        These scores evaluate performance on the held-out test data.
        Actual prices may differ due to vehicle condition, brand,
        model, location, and market demand.
        """
    )

# --------------------------------------------------
# 12. RUN THE APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.launch()
