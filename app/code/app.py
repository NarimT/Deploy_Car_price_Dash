# app.py — Car Price Prediction Dashboard
import json
import os
import math
import joblib
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# --------- Config ----------
MODEL_PATH = "models/car_price_pipeline.joblib"
META_PATH = "models/model_meta.json"   # optional

# --------- Load model & metadata ----------
model = None
model_error = None
meta = {}
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    model_error = str(e)

if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

expected_features = meta.get("features", ["year", "max_power"])
target_transform = meta.get("target_transform", "log")
assume_log_target = (target_transform == "log")

# --------- Small helpers ----------
def parse_mileage_to_number(s):
    """Accepts '45k', '45,000', '45000' and returns float or np.nan"""
    try:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return np.nan
        x = str(s).lower().replace(",", "").strip()
        if x.endswith("k"):
            return float(x[:-1]) * 1000.0
        return float(x)
    except Exception:
        return np.nan

def parse_owner(o):
    """Accept owner strings or ints -> return int or nan"""
    try:
        if o is None:
            return np.nan
        if isinstance(o, (int, float)) and not np.isnan(o):
            return int(o)
        s = str(o).strip().lower()
        mapping = {
            "first owner": 1, "second owner": 2, "third owner": 3,
            "fourth & above owner": 4, "fourth above owner": 4
        }
        if s.isdigit():
            return int(s)
        return mapping.get(s, np.nan)
    except Exception:
        return np.nan

# --------- Dash app ----------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        html.H2("Car Price Predictor", style={"textAlign": "center", "marginTop": 18}),
        html.P(
            ("Model: " + (MODEL_PATH if model is not None else "(not loaded)")),
            style={"textAlign": "center"},
        ),

        # Inputs: two columns
        dbc.Row([
            dbc.Col([
                dbc.Label("Year"),
                dbc.Input(id="input_year", type="number", placeholder="e.g. 2014", value=2014),
                html.Br(),
                dbc.Label("Max power (hp)"),
                dbc.Input(id="input_power", type="number", placeholder="e.g. 100", value=100),
                html.Br(),
                dbc.Label("Mileage (e.g. 45000 or 45k)"),
                dbc.Input(id="input_mileage", type="text", placeholder="Mileage", value=""),
            ], md=6),

            dbc.Col([
                dbc.Label("Engine (cc)"),
                dbc.Input(id="input_engine", type="number", placeholder="e.g. 1197", value=""),
                html.Br(),
                dbc.Label("Owner"),
                dcc.Dropdown(
                    id="input_owner",
                    options=[
                        {"label":"First Owner (1)","value":1},
                        {"label":"Second Owner (2)","value":2},
                        {"label":"Third Owner (3)","value":3},
                        {"label":"Fourth & Above (4)","value":4},
                    ],
                    value=1,
                    clearable=False
                ),
                html.Br(),
                dbc.Label("Brand (text)"),
                dbc.Input(id="input_brand", type="text", placeholder="e.g. Toyota", value=""),
                html.Br(),
                dbc.Label("Fuel"),
                dcc.Dropdown(
                    id="input_fuel",
                    options=[
                        {"label":"Petrol","value":"Petrol"},
                        {"label":"Diesel","value":"Diesel"},
                    ],
                    value="Petrol",
                    clearable=False
                ),
            ], md=6)
        ], className="mb-3"),

        dbc.Row(dbc.Col(dbc.Button("Predict", id="predict_btn", color="primary", className="w-100")), className="mb-3"),

        html.H4("Prediction"),
        html.Div(id="prediction_text", style={"fontSize": 20, "fontWeight": 600}),
        html.Hr(),
        html.H6("Debug / model info"),
        html.Pre(id="debug_area", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "color":"gray"})
    ],
    fluid=True,
    style={"maxWidth":"900px", "marginTop":"20px"}
)

# --------- Build input row ----------
def build_input_row_from_ui(year, power, mileage, engine, owner, brand, fuel, expected):
    # Validate year and power first
    try:
        y = int(year)
    except Exception:
        return None, "Year must be an integer (e.g. 2014)."
    try:
        p = float(power)
    except Exception:
        return None, "Max power must be a number (e.g. 100)."
    # Basic sanity
    if y < 1900 or y > 2030:
        return None, "Year looks out of range (1900-2030)."
    if not (0 < p < 10000):
        return None, "Max power looks out of range (0-10000)."

    # Parse other fields
    mil = parse_mileage_to_number(mileage)
    eng = None
    try:
        eng = float(engine) if engine not in (None, "", " ") else np.nan
    except Exception:
        eng = np.nan
    own = parse_owner(owner)
    brand_val = brand.strip() if isinstance(brand, str) and brand.strip() else np.nan
    fuel_val = fuel if fuel is not None else np.nan

    row = {}
    for f in expected:
        if f == "year":
            row["year"] = y
        elif f == "max_power":
            row["max_power"] = p
        elif f == "mileage":
            row["mileage"] = mil
        elif f == "engine":
            row["engine"] = eng
        elif f == "owner":
            row["owner"] = own
        elif f == "brand":
            row["brand"] = brand_val
        elif f == "fuel":
            row["fuel"] = fuel_val
        else:
            # unknown expected feature -> NaN (let pipeline imputer handle)
            row[f] = np.nan

    return pd.DataFrame([row]), None

# --------- Callback ----------
@app.callback(
    Output("prediction_text", "children"),
    Output("debug_area", "children"),
    Input("predict_btn", "n_clicks"),
    State("input_year", "value"),
    State("input_power", "value"),
    State("input_mileage", "value"),
    State("input_engine", "value"),
    State("input_owner", "value"),
    State("input_brand", "value"),
    State("input_fuel", "value"),
)
def on_predict(n_clicks, year, power, mileage, engine, owner, brand, fuel):
    debug_lines = []
    if model is None:
        debug_lines.append(f"Model not loaded: {model_error}")
        return "Model not loaded — see debug.", "\n".join(debug_lines)

    debug_lines.append(f"Model loaded from: {MODEL_PATH}")
    debug_lines.append(f"Expected features: {expected_features}")
    debug_lines.append(f"Assume log-target: {assume_log_target}")

    if not n_clicks:
        debug_lines.append("Waiting for Predict click — current sample row below (no prediction yet).")
        sample, _ = build_input_row_from_ui(year or 2014, power or 100, mileage or "", engine or "", owner or 1, brand or "", fuel or "Petrol", expected_features)
        debug_lines.append(sample.to_string(index=False))
        return "", "\n".join(debug_lines)

    X_row, err = build_input_row_from_ui(year, power, mileage, engine, owner, brand, fuel, expected_features)
    if err:
        debug_lines.append(f"Input validation error: {err}")
        return err, "\n".join(debug_lines)

    debug_lines.append("Input sent to model:")
    debug_lines.append(X_row.to_string(index=False))

    try:
        pred_raw = model.predict(X_row)
    except Exception as e:
        debug_lines.append("Prediction error (exception):")
        debug_lines.append(str(e))
        return "Prediction failed — see debug.", "\n".join(debug_lines)

    debug_lines.append(f"Raw model output: {repr(pred_raw)}")

    try:
        val = float(pred_raw[0])
    except Exception:
        debug_lines.append("Model output could not be parsed as float.")
        return "Prediction failed (unexpected model output).", "\n".join(debug_lines)

    if assume_log_target:
        if val > 50:
            debug_lines.append("Model output very large; avoid exp() to prevent overflow.")
            predicted_price = math.inf
        else:
            predicted_price = float(np.exp(val))
    else:
        predicted_price = val

    if not math.isfinite(predicted_price):
        debug_lines.append("Predicted price not finite.")
        return "Prediction produced non-finite value.", "\n".join(debug_lines)

    pred_str = f"Predicted selling price: {predicted_price:,.0f} INR"
    debug_lines.append(f"Interpreted predicted price: {predicted_price}")

    return pred_str, "\n".join(debug_lines)

# --------- Run ----------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
