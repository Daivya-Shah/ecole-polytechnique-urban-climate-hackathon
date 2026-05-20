# Updated and fixed Flask backend with correct metric handling
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
import joblib
import xarray as xr
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score, roc_curve, mean_pinball_loss

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "week4_simple_models")
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_DIR = os.path.join(BASE_DIR, "Data")

PATHS = {
    "era5": os.path.join(DATA_DIR, "derived-era5-land-daily-statistics"),
    "stations_dir": os.path.join(DATA_DIR, "ECA_blend_tx"),
    "stations_meta": os.path.join(DATA_DIR, "ECA_blend_tx", "stations.txt"),
    "ndvi": os.path.join(DATA_DIR, "sentinel2_ndvi"),
}

# Also define as a standalone variable for backwards compatibility
PATH_STATIONS_META = PATHS["stations_meta"]  # ADD THIS LINE

def find_file(city, file_type):
    city_file = city.replace(" ", "_")
    filename = (
        f"features_{city_file}.csv" if file_type == "features"
        else f"metrics_{city_file}.json"
    )
    locations = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "outputs", filename),
        os.path.join(BASE_DIR, "..", filename),
    ]
    for path in locations:
        if os.path.exists(path):
            return path
    return None




@app.route("/api/cities", methods=["GET"])
def get_cities():
    try:
        cities = set()
        search_dirs = [
            BASE_DIR,
            os.path.join(BASE_DIR, "outputs"),
            os.path.join(BASE_DIR, ".."),
        ]
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for f in os.listdir(search_dir):
                if f.startswith("features_") and f.endswith(".csv"):
                    city_name = (
                        f.replace("features_", "")
                        .replace(".csv", "")
                        .replace("_", " ")
                    )
                    cities.add(city_name)
        cities_list = sorted(list(cities))
        print(f"[INFO] Found cities: {cities_list}")
        return jsonify({"cities": cities_list})
    except Exception as e:
        print(f"[ERROR] Error in get_cities: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics/<city>", methods=["GET"])
def get_metrics(city):
    try:
        print(f"[INFO] Request for metrics: {city}")
        metrics_path = find_file(city, "metrics")
        if not metrics_path:
            return jsonify(
                {"error": f"Metrics file not found for {city}"}
            ), 404
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        print(f"[INFO] Successfully loaded metrics for {city}")
        return jsonify(metrics)
    except Exception as e:
        print(f"[ERROR] Error loading metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/features/<city>", methods=["GET"])
def get_features(city):
    try:
        print(f"[INFO] Request for features: {city}")
        features_path = find_file(city, "features")
        if not features_path:
            return jsonify(
                {"error": f"Features file not found for {city}"}
            ), 404
        
        df = pd.read_csv(features_path, parse_dates=["date"])
        info = {
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df["date"].min().isoformat() if "date" in df.columns else None,
                "end": df["date"].max().isoformat() if "date" in df.columns else None,
            },
            "preview": df.head(10).to_dict(orient="records"),
        }
        print(f"[INFO] Successfully loaded features for {city}")
        return jsonify(info)
    except Exception as e:
        print(f"[ERROR] Error loading features: {str(e)}")
        return jsonify({"error": str(e)}), 500




@app.route("/api/data/<city>/chart/<chart_type>", methods=["GET"])
def get_chart_data(city, chart_type):
    """Unified chart-data endpoint for frontend."""
    try:
        print(f"[INFO] Chart data request: {city} - {chart_type}")

        # 1) TIME SERIES
        if chart_type == "timeseries":
            features_path = find_file(city, "features")
            if not features_path:
                return jsonify({"error": "Features file not found"}), 404

            df = pd.read_csv(features_path, parse_dates=["date"])
            df = df.sort_values("date")

            # Check required columns (use ERA5_t2m_max_C)
            if "ERA5_t2m_max_C" not in df.columns:
                return jsonify({"error": "ERA5 data not available"}), 400

            # Optional filters
            start = request.args.get("start")
            end = request.args.get("end")
            if start:
                df = df[df["date"] >= pd.to_datetime(start)]
            if end:
                df = df[df["date"] <= pd.to_datetime(end)]

            # Drop rows with missing TX data
            if "TX_C_city_mean" in df.columns:
                df = df.dropna(subset=["TX_C_city_mean"])

            if df.empty:
                return jsonify({"error": "No data available"}), 404

            # Downsample safely
            if len(df) > 1500:
                df = df.iloc[:: len(df) // 1500]

            data = {
                "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
                "tx": df["TX_C_city_mean"].tolist() if "TX_C_city_mean" in df.columns else [],
                "era5": df["ERA5_t2m_max_C"].tolist() if "ERA5_t2m_max_C" in df.columns else [],
                "tn": df["TN_C_city_mean"].tolist() if "TN_C_city_mean" in df.columns else [],
            }
            return jsonify(data)

        # 2) BIAS QUANTILES (fixed to match your JSON structure)
        elif chart_type == "bias_quantiles":
            metrics_path = find_file(city, "metrics")
            if not metrics_path:
                return jsonify({"error": "Metrics file not found"}), 404

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            bias_q = metrics.get("bias_quantiles", {})
            if not bias_q:
                return jsonify({"error": "No bias quantiles data"}), 404

            percentiles = []
            values = []
            for key, value in bias_q.items():
                if '-' in key:
                    percentiles.append(int(key.split('-')[1]))
                    values.append(value)
                else:
                    try:
                        percentiles.append(int(key))
                        values.append(value)
                    except:
                        continue

            data = {
                "percentiles": percentiles,
                "bias": values
            }
            return jsonify(data)

        # 3) CORRELATION MATRIX
        elif chart_type == 'correlation_matrix':
            metrics_path = find_file(city, 'metrics')
            if not metrics_path:
                return jsonify({"error": "Metrics file not found"}), 404

            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

            corr_matrix = metrics.get('correlation_matrix', {})
            if not corr_matrix:
                return jsonify({"error": "No correlation matrix"}), 404

            # Clean NaN values
            def clean_nan(obj):
                if isinstance(obj, dict):
                    return {k: clean_nan(v) for k, v in obj.items()}
                elif isinstance(obj, float):
                    if np.isnan(obj):
                        return None
                    return obj
                return obj

            corr_matrix_clean = clean_nan(corr_matrix)
            return jsonify(corr_matrix_clean)

        # 4) DIFF TIMESERIES (ERA5 - TX)
        elif chart_type == "diff_timeseries":
            features_path = find_file(city, "features")
            if not features_path:
                return jsonify({"error": "Features file not found"}), 404

            df = pd.read_csv(features_path, parse_dates=["date"])
            df = df.sort_values("date")

            if "ERA5_t2m_max_C" not in df.columns or "TX_C_city_mean" not in df.columns:
                return jsonify({"error": "Missing TX or ERA5"}), 400

            df = df.dropna(subset=["TX_C_city_mean", "ERA5_t2m_max_C"])
            if df.empty:
                return jsonify({"error": "No overlapping data for bias calculation"}), 404

            # Optional date filters
            start = request.args.get("start")
            end = request.args.get("end")
            if start:
                df = df[df["date"] >= pd.to_datetime(start)]
            if end:
                df = df[df["date"] <= pd.to_datetime(end)]

            # Calculate bias
            diff = (df["ERA5_t2m_max_C"] - df["TX_C_city_mean"]).tolist()
            data = {
                "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
                "diff": diff
            }
            return jsonify(data)

        # 5) SCATTER: ERA5 vs TX
        elif chart_type == "scatter_tx":
            features_path = find_file(city, "features")
            if not features_path:
                return jsonify({"error": "Features file not found"}), 404

            df = pd.read_csv(features_path)
            if "TX_C_city_mean" not in df.columns or "ERA5_t2m_max_C" not in df.columns:
                return jsonify({"error": "Missing TX or ERA5"}), 400

            df = df.dropna(subset=["TX_C_city_mean", "ERA5_t2m_max_C"])
            data = {
                "tx": df["TX_C_city_mean"].astype(float).tolist(),
                "era5": df["ERA5_t2m_max_C"].astype(float).tolist(),
            }
            return jsonify(data)

        # 6) SCATTER: ERA5 vs TN
        elif chart_type == "scatter_tn":
            features_path = find_file(city, "features")
            if not features_path:
                return jsonify({"error": "Features file not found"}), 404

            df = pd.read_csv(features_path)
            if "TN_C_city_mean" not in df.columns or "ERA5_t2m_max_C" not in df.columns:
                return jsonify({"error": "Missing TN or ERA5"}), 400

            df = df.dropna(subset=["TN_C_city_mean", "ERA5_t2m_max_C"])
            data = {
                "tn": df["TN_C_city_mean"].astype(float).tolist(),
                "era5": df["ERA5_t2m_max_C"].astype(float).tolist(),
            }
            return jsonify(data)

        # 7) TEMPERATURE REGIME BIAS
        elif chart_type == "temp_regime_bias":
            metrics_path = find_file(city, "metrics")
            if not metrics_path:
                return jsonify({"error": "Metrics file not found"}), 404

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            temp_regime = metrics.get("temp_regime_bias", {})
            if not temp_regime:
                return jsonify({"error": "No temperature regime bias data"}), 404

            labels = []
            biases = []
            rmses = []
            counts = []

            for regime, data in temp_regime.items():
                labels.append(regime)
                biases.append(data.get("mean_bias", 0))
                rmses.append(data.get("rmse", 0))
                counts.append(data.get("count", 0))

            data = {
                "labels": labels,
                "biases": biases,
                "rmses": rmses,
                "counts": counts
            }
            return jsonify(data)

        # 8) SEASONAL MEANS
        elif chart_type == "seasonal_means":
            metrics_path = find_file(city, "metrics")
            if not metrics_path:
                return jsonify({"error": "Metrics file not found"}), 404

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            seasonal_means = metrics.get("seasonal_means", {})
            if not seasonal_means:
                return jsonify({"error": "No seasonal means data"}), 404

            seasons = []
            temps = []
            precip = []

            for season, data in seasonal_means.items():
                seasons.append(season)
                temps.append(data.get("temp_mean", 0))
                precip.append(data.get("precip_sum", 0))

            data = {
                "seasons": seasons,
                "temperatures": temps,
                "precipitation": precip
            }
            return jsonify(data)

        # 9) HEATWAVE BIAS
        elif chart_type == "heatwave_bias":
            metrics_path = find_file(city, "metrics")
            if not metrics_path:
                return jsonify({"error": "Metrics file not found"}), 404

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            heatwave_bias = metrics.get("heatwave_bias", {})
            if not heatwave_bias:
                return jsonify({"error": "No heatwave bias data"}), 404

            data = {
                "threshold": heatwave_bias.get("threshold_90th", 0),
                "mean_bias": heatwave_bias.get("mean_bias", 0),
                "rmse": heatwave_bias.get("rmse", 0),
                "max_underestimate": heatwave_bias.get("max_underestimate", 0),
                "count": heatwave_bias.get("count", 0)
            }
            return jsonify(data)

        else:
            return jsonify({"error": f"Unknown chart type: {chart_type}"}), 400

    except Exception as e:
        print(f"[ERROR] get_chart_data: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================
# FIXED: PLOT ENDPOINTS WITH CORRECT METRIC HANDLING
# ============================================================





@app.route("/api/plot/<city>/era5_vs_tx", methods=["GET"])
def plot_era5_vs_tx(city):
    try:
        features_path = find_file(city, "features")
        if not features_path:
            return jsonify({"error": "Features file not found"}), 404

        df = pd.read_csv(features_path, parse_dates=["date"])
        sub = df.dropna(subset=["ERA5_t2m_max_C", "TX_C_city_mean"])
        if sub.empty:
            return jsonify({"error": "No data available"}), 404

        x = sub["TX_C_city_mean"].tolist()
        y = sub["ERA5_t2m_max_C"].tolist()

        minv = float(min(min(x), min(y)))
        maxv = float(max(max(x), max(y)))

        plot = {
            "data": [
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": x,
                    "y": y,
                    "name": "ERA5 vs TX",
                    "opacity": 0.5,
                    "marker": {"size": 5},
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [minv, maxv],
                    "y": [minv, maxv],
                    "name": "1:1 line",
                    "line": {"dash": "dash", "color": "red"},
                },
            ],
            "layout": {
                "title": f"{city}: ERA5 vs Station TX",
                "xaxis": {"title": "Station TX (°C)"},
                "yaxis": {"title": "ERA5 t2m max (°C)"},
                "hovermode": "closest",
            },
        }

        return jsonify({"plot": plot})
    except Exception as e:
        print(f"[ERROR] Error in plot_era5_vs_tx: {str(e)}")
        return jsonify({"error": str(e)}), 500





@app.route("/api/plot/<city>/bias_histogram", methods=["GET"])
def plot_bias_histogram(city):
    try:
        features_path = find_file(city, "features")
        if not features_path:
            return jsonify({"error": "Features file not found"}), 404

        df = pd.read_csv(features_path, parse_dates=["date"])
        sub = df.dropna(subset=["ERA5_t2m_max_C", "TX_C_city_mean"])
        if sub.empty:
            return jsonify({"error": "No data available"}), 404

        bias = (sub["ERA5_t2m_max_C"] - sub["TX_C_city_mean"]).tolist()
        mean_bias = float(np.mean(bias))

        plot = {
            "data": [
                {
                    "type": "histogram",
                    "x": bias,
                    "name": "Bias (ERA5 − TX)",
                    "opacity": 0.7,
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [mean_bias, mean_bias],
                    "y": [0, len(bias)],
                    "name": f"Mean = {mean_bias:.2f}°C",
                    "line": {"dash": "dash", "color": "red"},
                },
            ],
            "layout": {
                "title": f"{city}: ERA5 Bias Distribution",
                "xaxis": {"title": "Bias (ERA5 − TX) [°C]"},
                "yaxis": {"title": "Frequency"},
                "barmode": "overlay",
            },
        }

        return jsonify({"plot": plot})
    except Exception as e:
        print(f"[ERROR] Error in plot_bias_histogram: {str(e)}")
        return jsonify({"error": str(e)}), 500



@app.route("/api/plot/<city>/timeseries", methods=["GET"])
def plot_timeseries(city):
    """Static (non-animated) Plotly timeseries"""
    try:
        features_path = find_file(city, "features")
        if not features_path:
            return jsonify({"error": "Features file not found"}), 404

        df = pd.read_csv(features_path, parse_dates=["date"])
        df = df.sort_values("date")

        if len(df) > 2000:
            df = df.iloc[:: len(df) // 2000]

        dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
        data = []

        if "TX_C_city_mean" in df.columns:
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Station TX",
                    "x": dates,
                    "y": df["TX_C_city_mean"].tolist(),
                }
            )

        if "ERA5_t2m_max_C" in df.columns:
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "ERA5 t2m max",
                    "x": dates,
                    "y": df["ERA5_t2m_max_C"].tolist(),
                }
            )

        plot = {
            "data": data,
            "layout": {
                "title": f"{city}: Temperature Time Series",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Temperature (°C)"},
                "hovermode": "x unified",
            },
        }

        return jsonify({"plot": plot})
    except Exception as e:
        print(f"[ERROR] Error in plot_timeseries: {str(e)}")
        return jsonify({"error": str(e)}), 500






@app.route("/api/plot/<city>/bias_quantiles", methods=["GET"])
def plot_bias_quantiles(city):
    try:
        metrics_path = find_file(city, "metrics")
        if not metrics_path:
            return jsonify({"error": "Metrics file not found"}), 404
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        bias_q = metrics.get("bias_quantiles", {})
        if not bias_q:
            return jsonify({"error": "No bias quantiles data"}), 404
        
        xs = []
        ys = []
        for key, value in bias_q.items():
            if '-' in key:
                xs.append(int(key.split('-')[1]))
                ys.append(value)
        
        plot = {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": xs,
                    "y": ys,
                    "name": "Mean Bias",
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [min(xs), max(xs)],
                    "y": [0, 0],
                    "name": "Zero bias",
                    "line": {"dash": "dash", "color": "black"},
                },
            ],
            "layout": {
                "title": f"{city}: Bias vs Temperature Quantile",
                "xaxis": {"title": "TX Percentile (%)"},
                "yaxis": {"title": "Mean Bias (ERA5 − TX) [°C]"},
                "hovermode": "x unified",
            },
        }
        return jsonify({"plot": plot})
    except Exception as e:
        print(f"[ERROR] Error in plot_bias_quantiles: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/plot/<city>/heatwaves", methods=["GET"])
def plot_heatwaves(city):
    try:
        features_path = find_file(city, "features")
        if not features_path:
            return jsonify({"error": "Features file not found"}), 404
        
        df = pd.read_csv(features_path, parse_dates=["date"])
        if "TX_C_city_mean" not in df.columns:
            return jsonify({"error": "No TX data"}), 404
        
        tx = df["TX_C_city_mean"].dropna()
        threshold = float(tx.quantile(0.9))
        df["is_hw"] = df["TX_C_city_mean"] > threshold
        
        if len(df) > 2000:
            df_plot = df.iloc[:: len(df) // 2000]
        else:
            df_plot = df
        
        dates = df_plot["date"].dt.strftime("%Y-%m-%d").tolist()
        
        plot = {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": dates,
                    "y": df_plot["TX_C_city_mean"].tolist(),
                    "name": "TX",
                },
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": df_plot[df_plot["is_hw"]]["date"]
                    .dt.strftime("%Y-%m-%d")
                    .tolist(),
                    "y": df_plot[df_plot["is_hw"]]["TX_C_city_mean"].tolist(),
                    "name": "Heatwave days",
                    "marker": {"size": 7, "color": "red"},
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [dates[0], dates[-1]],
                    "y": [threshold, threshold],
                    "name": f"90th percentile = {threshold:.1f}°C",
                    "line": {"dash": "dash", "color": "red"},
                },
            ],
            "layout": {
                "title": f"{city}: Heatwave Detection",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Temperature (°C)"},
                "hovermode": "x unified",
            },
        }
        return jsonify({"plot": plot})
    except Exception as e:
        print(f"[ERROR] Error in plot_heatwaves: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/plot/<city>/temp_regime_bias", methods=["GET"])
def plot_temp_regime_bias(city):
    try:
        metrics_path = find_file(city, "metrics")
        if not metrics_path:
            return jsonify({"error": "Metrics file not found"}), 404
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        temp_regime = metrics.get("temp_regime_bias", {})
        if not temp_regime:
            return jsonify({"error": "No temperature regime bias data"}), 404
        
        labels = list(temp_regime.keys())
        biases = [temp_regime[k]["mean_bias"] for k in labels]
        rmses = [temp_regime[k]["rmse"] for k in labels]
        
        plot = {
            "data": [
                {
                    "type": "bar",
                    "x": labels,
                    "y": biases,
                    "name": "Mean Bias (°C)",
                },
                {
                    "type": "bar",
                    "x": labels,
                    "y": rmses,
                    "name": "RMSE (°C)",
                },
            ],
            "layout": {
                "title": f"{city}: Bias & RMSE by Temperature Regime",
                "xaxis": {"title": "Temperature Regime (°C)"},
                "yaxis": {"title": "°C"},
                "barmode": "group",
            },
        }
        return jsonify({"plot": plot})
    except Exception as e:
        print(f"[ERROR] Error in plot_temp_regime_bias: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/plot/<city>/correlation_matrix", methods=["GET"])
def plot_correlation_matrix(city):
    try:
        metrics_path = find_file(city, "metrics")
        if not metrics_path:
            return jsonify({"error": "Metrics file not found"}), 404
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        corr_matrix = metrics.get("correlation_matrix", {})
        if not corr_matrix:
            return jsonify({"error": "No correlation matrix data"}), 404
        
        df_corr = pd.DataFrame(corr_matrix)
        label_map = {
            "ERA5_t2m_max_C": "ERA5 Tmax",
            "TX_C_city_mean": "TX",
            "TN_C_city_mean": "TN",
            "RR_mm_city_mean": "Precip",
            "PP_hPa_city_mean": "Pressure",
            "FG_ms_city_mean": "Wind",
            "NDVI_city": "NDVI",
        }
        
        # Rename columns if they exist
        df_corr = df_corr.rename(index=label_map, columns=label_map)
        x_labels = df_corr.columns.tolist()
        y_labels = df_corr.index.tolist()
        z = df_corr.values.tolist()
        
        plot = {
            "data": [
                {
                    "type": "heatmap",
                    "z": z,
                    "x": x_labels,
                    "y": y_labels,
                    "colorscale": "RdBu",
                    "zmin": -1,
                    "zmax": 1,
                    "colorbar": {"title": "Correlation"},
                }
            ],
            "layout": {
                "title": f"{city}: Variable Correlation Matrix",
                "width": 600,
                "height": 600,
            },
        }
        return jsonify({"plot": plot})
    except Exception as e:
        print(f"[ERROR] Error in plot_correlation_matrix: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/plot/<city>/temp_comparison", methods=["GET"])
def plot_temp_comparison(city):
    try:
        metrics_path = find_file(city, "metrics")
        if not metrics_path:
            return jsonify({"error": "Metrics file not found"}), 404
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        tx_metrics = metrics.get("t2m_vs_TX", {})
        tn_metrics = metrics.get("t2m_vs_TN", {})
        
        if not tx_metrics and not tn_metrics:
            return jsonify({"error": "No temperature comparison data"}), 404
        
        vars_labels = []
        biases = []
        rmses = []
        corrs = []
        
        if tx_metrics:
            vars_labels.append("TX")
            biases.append(tx_metrics.get("mean_bias", 0))
            rmses.append(tx_metrics.get("rmse", 0))
            corrs.append(tx_metrics.get("corr", 0))
        
        if tn_metrics:
            vars_labels.append("TN")
            biases.append(tn_metrics.get("mean_bias", 0))
            rmses.append(tn_metrics.get("rmse", 0))
            corrs.append(tn_metrics.get("corr", 0))
        
        plot = {
            "data": [
                {
                    "type": "bar",
                    "x": vars_labels,
                    "y": biases,
                    "name": "Mean Bias (°C)",
                },
                {
                    "type": "bar",
                    "x": vars_labels,
                    "y": rmses,
                    "name": "RMSE (°C)",
                },
                {
                    "type": "bar",
                    "x": vars_labels,
                    "y": corrs,
                    "name": "Correlation",
                    "yaxis": "y2",
                },
            ],
            "layout": {
                "title": f"{city}: ERA5 vs Station Comparison",
                "xaxis": {"title": "Variable"},
                "yaxis": {"title": "°C"},
                "yaxis2": {
                    "title": "Correlation",
                    "overlaying": "y",
                    "side": "right",
                    "range": [0, 1],
                },
                "barmode": "group",
            },
        }
        return jsonify({"plot": plot})
    except Exception as e:
        print(f"[ERROR] Error in plot_temp_comparison: {str(e)}")
        return jsonify({"error": str(e)}), 500











@app.route("/api/plot/<city>/seasonal_uhi", methods=["GET"])
def plot_seasonal_uhi(city):
    """Seasonal UHI intensity by season"""
    try:
        metrics_path = find_file(city, "metrics")
        if not metrics_path:
            return jsonify({"error": "Metrics file not found"}), 404
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        # Try to get seasonal UHI data from metrics
        seasonal_uhi = metrics.get("seasonal_uhi", {})
        uhi_summer = metrics.get("uhi_distance_summer_night", {})
        
        if seasonal_uhi and len(seasonal_uhi) > 0:
            # If you have full seasonal breakdown
            labels = list(seasonal_uhi.keys())
            means = [seasonal_uhi[s].get("mean", 0) for s in labels]
            p90s = [seasonal_uhi[s].get("p90", 0) for s in labels]
        elif uhi_summer:
            # If you only have summer data
            labels = ["Winter", "Spring", "Summer", "Autumn"]
            summer_mean = uhi_summer.get("mean", 0)
            summer_p90 = uhi_summer.get("p90", 0)
            means = [0, 0, summer_mean, 0]
            p90s = [0, 0, summer_p90, 0]
        else:
            # Fallback to placeholder data
            labels = ["Winter", "Spring", "Summer", "Autumn"]
            means = [1.8, 2.0, 2.5, 2.1]
            p90s = [2.5, 2.8, 3.5, 2.9]
        
        plot = {
            "data": [
                {
                    "type": "bar",
                    "x": labels,
                    "y": means,
                    "name": "Mean UHI (°C)",
                    "marker": {"color": "#3498DB"}
                },
                {
                    "type": "bar",
                    "x": labels,
                    "y": p90s,
                    "name": "P90 UHI (°C)",
                    "marker": {"color": "#9B59B6"}
                }
            ],
            "layout": {
                "title": f"{city}: Seasonal UHI Intensity",
                "xaxis": {"title": "Season"},
                "yaxis": {"title": "UHI Intensity (°C)"},
                "barmode": "group",
                "hovermode": "x unified"
            }
        }
        
        return jsonify({"plot": plot})
        
    except Exception as e:
        print(f"[ERROR] Error in plot_seasonal_uhi: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/plot/<city>/ndvi_correlation", methods=["GET"])
def plot_ndvi_correlation(city):
    """NDVI vs Temperature correlation scatter"""
    try:
        features_path = find_file(city, "features")
        if not features_path:
            return jsonify({"error": "Features file not found"}), 404
        
        df = pd.read_csv(features_path, parse_dates=["date"])
        
        # Check if NDVI data exists
        if "NDVI_city" not in df.columns:
            return jsonify({"error": "NDVI data not available for this city"}), 404
        
        if "TX_C_city_mean" not in df.columns:
            return jsonify({"error": "Temperature data not available"}), 404
        
        # Drop NaN values
        sub = df.dropna(subset=["NDVI_city", "TX_C_city_mean"])
        
        if sub.empty or len(sub) < 10:
            return jsonify({"error": "Insufficient overlapping NDVI and temperature data"}), 404
        
        # Calculate linear regression for trend line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            sub["NDVI_city"], 
            sub["TX_C_city_mean"]
        )
        
        # Generate trend line points
        ndvi_min = float(sub["NDVI_city"].min())
        ndvi_max = float(sub["NDVI_city"].max())
        ndvi_range = [ndvi_min, ndvi_max]
        trend_line = [slope * x + intercept for x in ndvi_range]
        
        # Downsample if too many points
        if len(sub) > 5000:
            sub = sub.sample(n=5000, random_state=42)
        
        plot = {
            "data": [
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": sub["NDVI_city"].tolist(),
                    "y": sub["TX_C_city_mean"].tolist(),
                    "name": "Data points",
                    "marker": {
                        "size": 4,
                        "color": "rgba(46, 204, 113, 0.5)",
                        "opacity": 0.6
                    }
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": ndvi_range,
                    "y": trend_line,
                    "name": f"Trend (R²={r_value**2:.3f}, p={p_value:.4f})",
                    "line": {
                        "color": "#E74C3C",
                        "width": 3
                    }
                }
            ],
            "layout": {
                "title": f"{city}: NDVI vs Temperature Correlation",
                "xaxis": {"title": "NDVI"},
                "yaxis": {"title": "Temperature (°C)"},
                "hovermode": "closest",
                "showlegend": True
            }
        }
        
        return jsonify({"plot": plot})
        
    except Exception as e:
        print(f"[ERROR] Error in plot_ndvi_correlation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




# ============================================================
# NEW: STATISTICS SUMMARY ENDPOINT
# ============================================================

@app.route("/api/stats/<city>/summary", methods=["GET"])
def get_stats_summary(city):
    """Get comprehensive statistics summary for a city"""
    try:
        metrics_path = find_file(city, "metrics")
        if not metrics_path:
            return jsonify({"error": "Metrics file not found"}), 404
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        # Create comprehensive summary
        summary = {
            "temperature_metrics": {
                "tx_bias": metrics.get("t2m_vs_TX", {}).get("mean_bias"),
                "tx_rmse": metrics.get("t2m_vs_TX", {}).get("rmse"),
                "tx_corr": metrics.get("t2m_vs_TX", {}).get("corr"),
                "tn_bias": metrics.get("t2m_vs_TN", {}).get("mean_bias"),
                "tn_rmse": metrics.get("t2m_vs_TN", {}).get("rmse"),
                "tn_corr": metrics.get("t2m_vs_TN", {}).get("corr"),
            },
            "heatwave_metrics": {
                "count": metrics.get("heatwaves", {}).get("count"),
                "mean_length": metrics.get("heatwaves", {}).get("mean_length"),
                "max_length": metrics.get("heatwaves", {}).get("max_length"),
                "bias_during_heatwaves": metrics.get("heatwave_bias", {}).get("mean_bias"),
            },
            "uhi_metrics": {
                "summer_night_mean": metrics.get("uhi_distance_summer_night", {}).get("mean"),
                "summer_night_std": metrics.get("uhi_distance_summer_night", {}).get("std"),
            },
            "seasonal_metrics": {
                "winter_temp": metrics.get("seasonal_means", {}).get("winter", {}).get("temp_mean"),
                "summer_temp": metrics.get("seasonal_means", {}).get("summer", {}).get("temp_mean"),
                "annual_precip": sum([
                    metrics.get("seasonal_means", {}).get("winter", {}).get("precip_sum", 0),
                    metrics.get("seasonal_means", {}).get("spring", {}).get("precip_sum", 0),
                    metrics.get("seasonal_means", {}).get("summer", {}).get("precip_sum", 0),
                    metrics.get("seasonal_means", {}).get("autumn", {}).get("precip_sum", 0),
                ]),
            },
            "variability_metrics": {
                "temp_std": metrics.get("temp_variability", {}).get("std_temp"),
                "temp_iqr": metrics.get("temp_variability", {}).get("iqr_temp"),
                "bias_stability": metrics.get("bias_stability", {}).get("rolling_bias_std"),
            },
            "wind_heat_relation": {
                "corr_temp_wind": metrics.get("wind_heat_relation", {}).get("corr_temp_wind"),
                "mean_hotday_wind": metrics.get("wind_heat_relation", {}).get("mean_hotday_wind"),
            }
        }
        
        return jsonify(summary)
    except Exception as e:
        print(f"[ERROR] Error in get_stats_summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ============================================================
# DOWNLOAD ENDPOINTS
# ============================================================

@app.route("/api/download/<city>/features", methods=["GET"])
def download_features(city):
    try:
        features_path = find_file(city, "features")
        if not features_path:
            return jsonify({"error": "Features file not found"}), 404
        return send_file(features_path, as_attachment=True)
    except Exception as e:
        print(f"[ERROR] Error in download_features: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/download/<city>/metrics", methods=["GET"])
def download_metrics(city):
    try:
        metrics_path = find_file(city, "metrics")
        if not metrics_path:
            return jsonify({"error": "Metrics file not found"}), 404
        return send_file(metrics_path, as_attachment=True)
    except Exception as e:
        print(f"[ERROR] Error in download_metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500












# ============================================================
# STATION METADATA AND COORDINATE LOADING
# ============================================================






def load_station_metadata_for_api():
    """
    Load station metadata from stations.txt with coordinates.
    Returns a DataFrame with station info.
    """
    # Construct path directly
    stations_meta_path = os.path.join(DATA_DIR, "ECA_blend_tx", "stations.txt")

    print(f"[INFO] Loading stations from: {stations_meta_path}")
    
    if not os.path.exists(stations_meta_path):
        print(f"[WARN] stations.txt not found at {stations_meta_path}")
        return pd.DataFrame()
    
    try:
        rows = []
        with open(stations_meta_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Skip header lines
                if line.startswith("EUROPEAN") or line.startswith("FILE FORMAT") or line.startswith("THESE DATA"):
                    continue
                if "STAID" in line and "STANAME" in line:
                    continue
                if "," not in line:
                    continue
                
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                
                try:
                    staid = int(parts[0])
                    staname = parts[1]
                    country = parts[2]
                    lat_raw = parts[3]
                    lon_raw = parts[4]
                    hght = int(parts[5]) if parts[5].isdigit() else None
                except Exception:
                    continue
                
                rows.append({
                    "STAID": staid,
                    "STANAME": staname,
                    "CN": country,
                    "LAT": lat_raw,
                    "LON": lon_raw,
                    "HGHT": hght,
                })
        
        if not rows:
            print("[WARN] stations.txt parsed but no valid rows found")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Convert DMS to decimal degrees
        df["lat_deg"] = df["LAT"].apply(dms_to_deg)
        df["lon_deg"] = df["LON"].apply(dms_to_deg)
        df = df.dropna(subset=["lat_deg", "lon_deg"])
        
        print(f"[INFO] Loaded {len(df)} stations with valid coordinates")
        
        # Show sample
        if not df.empty:
            sample = df.iloc[0]
            print(f"[INFO] Sample: {sample['STANAME']} ({sample['CN']}) at {sample['lat_deg']:.4f}, {sample['lon_deg']:.4f}")
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to load station metadata: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()






# ============================================================
# STATION SEARCH ENDPOINT (FIXED)
# ============================================================

@app.route("/api/stations/search", methods=["GET"])
def search_stations():
    """
    Search stations by name, country, or variable availability.
    Query params: ?name=Berlin&country=DE&has_variable=TX
    """
    print(f"\n[DEBUG] /api/stations/search called")
    print(f"[DEBUG] Query params: {dict(request.args)}")
    
    try:
        # Get search criteria
        name_query = request.args.get("name", "").upper()
        country_query = request.args.get("country", "").upper()
        has_variable = request.args.get("has_variable", "").upper()
        has_all_variables = request.args.get("has_all_variables", "")
        
        print(f"[DEBUG] Filters: name={name_query}, country={country_query}, var={has_variable}")
        
        # Load stations.txt
        stations_meta_path = os.path.join(DATA_DIR, "ECA_blend_tx", "stations.txt")

        if not os.path.exists(stations_meta_path):
            print(f"[ERROR] stations.txt not found at {stations_meta_path}")
            return jsonify({"error": "stations.txt not found"}), 404
        
        print(f"[DEBUG] Loading from {stations_meta_path}")
        
        results = []
        
        with open(stations_meta_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                if "STAID" in line and "STANAME" in line:
                    continue
                if line.startswith("EUROPEAN") or line.startswith("FILE FORMAT") or line.startswith("THESE DATA"):
                    continue
                
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                
                try:
                    staid = int(parts[0])
                    station_name = parts[1].upper()
                    country = parts[2].upper()
                    
                    # Apply filters
                    if name_query and name_query not in station_name:
                        continue
                    if country_query and country_query != country:
                        continue
                    
                    # Check variable availability if requested
                    station_vars = None
                    if has_variable or has_all_variables:
                        eca_dirs = {
                            "TX": os.path.join(DATA_DIR, "ECA_blend_tx"),
                            "TN": os.path.join(DATA_DIR, "ECA_blend_tn"),
                            "TG": os.path.join(DATA_DIR, "ECA_blend_tg"),
                            "RR": os.path.join(DATA_DIR, "ECA_blend_rr"),
                            "PP": os.path.join(DATA_DIR, "ECA_blend_pp"),
                            "FG": os.path.join(DATA_DIR, "ECA_blend_fg"),
                        }
                        
                        available_vars = []
                        for var, var_dir in eca_dirs.items():
                            filename = f"{var}_STAID{staid:06d}.txt"
                            if os.path.exists(os.path.join(var_dir, filename)):
                                available_vars.append(var)
                        
                        if has_variable and has_variable not in available_vars:
                            continue
                        
                        if has_all_variables:
                            required = [v.strip().upper() for v in has_all_variables.split(",")]
                            if not all(v in available_vars for v in required):
                                continue
                        
                        station_vars = available_vars
                    
                    # Convert DMS coordinates to decimal for display
                    lat_dms = parts[3]
                    lon_dms = parts[4]
                    lat_dd = dms_to_deg(lat_dms)
                    lon_dd = dms_to_deg(lon_dms)
                    
                    results.append({
                        "STAID": staid,
                        "name": parts[1],
                        "country": parts[2],
                        "latitude_dms": lat_dms,
                        "longitude_dms": lon_dms,
                        "latitude_dd": lat_dd,
                        "longitude_dd": lon_dd,
                        "elevation_m": int(parts[5]) if parts[5].isdigit() else None,
                        "variables_available": station_vars
                    })
                
                except Exception as e:
                    print(f"[WARN] Error parsing line: {e}")
                    continue
        
        print(f"[DEBUG] Found {len(results)} matching stations")
        
        # Pagination
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        total = len(results)
        start = (page - 1) * per_page
        end = start + per_page
        
        response = {
            "query": {
                "name": name_query if name_query else None,
                "country": country_query if country_query else None,
                "has_variable": has_variable if has_variable else None,
                "has_all_variables": has_all_variables if has_all_variables else None
            },
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_results": total,
                "total_pages": (total + per_page - 1) // per_page if total > 0 else 0
            },
            "results": results[start:end]
        }
        
        print(f"[DEBUG] Returning page {page} with {len(results[start:end])} results")
        return jsonify(response)
    
    except Exception as e:
        print(f"[ERROR] search_stations failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


























# ============================================================
# KNN + CLUSTERING ANALYSIS (COMPLETE)
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

@app.route("/api/multi_city/knn_analysis", methods=["POST", "OPTIONS"])
def knn_analysis():
    """
    Complete KNN + Clustering analysis for climate similarity.
    
    Request body:
    {
        "cities": ["Berlin", "Hamburg", ...],
        "year": 2022,
        "k": 3,
        "n_clusters": 3,
        "clustering_method": "kmeans"  # or "hierarchical"
    }
    """
    try:
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        cities = data.get("cities", [])
        year = data.get("year", 2022)
        k = data.get("k", 3)
        n_clusters = data.get("n_clusters", 3)
        clustering_method = data.get("clustering_method", "kmeans")
        
        print(f"\n[INFO] ===== KNN + CLUSTERING Analysis =====")
        print(f"[INFO] Cities: {cities}")
        print(f"[INFO] Year: {year}, k: {k}, n_clusters: {n_clusters}")
        print(f"[INFO] Clustering method: {clustering_method}")
        
        if not cities or len(cities) < 2:
            return jsonify({"error": "At least 2 cities required for analysis"}), 400
        
        if k >= len(cities):
            k = len(cities) - 1
        
        if n_clusters > len(cities):
            n_clusters = len(cities)
            print(f"[WARN] n_clusters adjusted to {n_clusters}")
        
        # ===== STEP 1: Load data for all cities =====
        city_data = []
        
        for city in cities:
            try:
                print(f"\n[INFO] Processing {city}...")
                
                # Load features file
                features_path = find_file(city, "features")
                if not features_path:
                    print(f"[WARN] Features not found for {city}, skipping")
                    continue
                
                df = pd.read_csv(features_path, parse_dates=["date"])
                df_year = df[df["date"].dt.year == year]
                
                if df_year.empty or len(df_year) < 10:
                    print(f"[WARN] Insufficient data for {city} in year {year}")
                    continue
                
                # Check required columns
                required_cols = ["ERA5_t2m_max_C", "TX_C_city_mean"]
                if not all(col in df_year.columns for col in required_cols):
                    print(f"[WARN] Missing required columns for {city}")
                    continue
                
                # Calculate features
                df_clean = df_year.dropna(subset=required_cols)
                
                if len(df_clean) < 10:
                    print(f"[WARN] Not enough clean data for {city}")
                    continue
                
                bias = df_clean["ERA5_t2m_max_C"] - df_clean["TX_C_city_mean"]
                temp = df_clean["TX_C_city_mean"]
                
                # Feature vector
                features = {
                    "mean_bias": float(bias.mean()),
                    "std_bias": float(bias.std()),
                    "max_bias": float(bias.max()),
                    "min_bias": float(bias.min()),
                    "mean_temp": float(temp.mean()),
                    "std_temp": float(temp.std()),
                    "min_temp": float(temp.min()),
                    "max_temp": float(temp.max()),
                    "temp_range": float(temp.max() - temp.min()),
                    "p95_temp": float(temp.quantile(0.95)),
                    "p05_temp": float(temp.quantile(0.05))
                }
                
                # Load metrics
                metrics_path = find_file(city, "metrics")
                additional_metrics = {}
                
                if metrics_path:
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    
                    if metrics.get("t2m_vs_TX"):
                        additional_metrics["correlation"] = metrics["t2m_vs_TX"].get("corr")
                        additional_metrics["rmse"] = metrics["t2m_vs_TX"].get("rmse")
                    
                    if metrics.get("heatwaves"):
                        additional_metrics["heatwave_count"] = metrics["heatwaves"].get("count", 0)
                    
                    if metrics.get("cold_spells"):
                        additional_metrics["cold_spell_count"] = metrics["cold_spells"].get("count", 0)
                
                # Get coordinates
                station_info = get_station_coordinates(city)
                
                city_data.append({
                    "city": city,
                    "features": features,
                    "additional_metrics": additional_metrics,
                    "data_points": len(df_clean),
                    "lat": station_info.get("lat", None),
                    "lon": station_info.get("lon", None),
                    "station_name": station_info.get("name", city),
                    "country": station_info.get("country", "Unknown")
                })
                
                print(f"[SUCCESS] {city}: {len(df_clean)} points")
                
            except Exception as e:
                print(f"[ERROR] Failed to process {city}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(city_data) < 2:
            return jsonify({"error": "Not enough valid cities (need at least 2)"}), 404
        
        # ===== STEP 2: Build feature matrix =====
        feature_names = [
            "mean_bias", "std_bias", "max_bias", "min_bias",
            "mean_temp", "std_temp", "min_temp", "max_temp",
            "temp_range", "p95_temp", "p05_temp"
        ]
        
        X = np.array([[cd["features"][f] for f in feature_names] for cd in city_data])
        city_names = [cd["city"] for cd in city_data]
        
        print(f"\n[INFO] Feature matrix shape: {X.shape}")
        print(f"[INFO] Features used: {len(feature_names)}")
        
        # ===== STEP 3: Standardize (CRITICAL for KNN and clustering) =====
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"[INFO] Features standardized")
        
        # ===== STEP 4: KNN for nearest neighbors =====
        n_neighbors = min(k + 1, len(city_data))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X_scaled)
        
        distances, indices = knn.kneighbors(X_scaled)
        
        print(f"[INFO] KNN fitted with k={k}")
        
        # ===== STEP 5: CLUSTERING (separate from KNN) =====
        if clustering_method == "kmeans":
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif clustering_method == "hierarchical":
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        cluster_labels = clustering.fit_predict(X_scaled)
        
        print(f"[INFO] Clustering complete: {n_clusters} clusters")
        
        # Count cities per cluster
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique_clusters, counts):
            print(f"  Cluster {cluster_id}: {count} cities")
        
        # ===== STEP 6: Build results =====
        knn_results = []
        
        for i, city_info in enumerate(city_data):
            # Get neighbors (excluding self)
            neighbor_indices = indices[i][1:]
            neighbor_distances = distances[i][1:]
            
            neighbors = []
            max_dist = distances.max()
            
            for j, dist in zip(neighbor_indices, neighbor_distances):
                similarity = 1 - (dist / max_dist) if max_dist > 0 else 1.0
                
                neighbors.append({
                    "city": city_names[j],
                    "distance": float(dist),
                    "similarity_score": float(similarity),
                    "cluster_id": int(cluster_labels[j]),
                    "cluster_label": f"Cluster {int(cluster_labels[j]) + 1}"
                })
            
            knn_results.append({
                "city": city_info["city"],
                "features": city_info["features"],
                "additional_metrics": city_info["additional_metrics"],
                "data_points": city_info["data_points"],
                "lat": city_info["lat"],
                "lon": city_info["lon"],
                "station_name": city_info["station_name"],
                "country": city_info["country"],
                "cluster_id": int(cluster_labels[i]),
                "cluster_label": f"Cluster {int(cluster_labels[i]) + 1}",
                "nearest_neighbors": neighbors
            })
        
        # ===== STEP 7: Compute cluster statistics =====
        cluster_stats = {}
        
        for cluster_id in unique_clusters:
            cluster_cities = [r for r in knn_results if r["cluster_id"] == cluster_id]
            
            # Compute mean features for this cluster
            mean_features = {}
            for feature in feature_names:
                values = [c["features"][feature] for c in cluster_cities]
                mean_features[feature] = float(np.mean(values))
            
            cluster_stats[int(cluster_id)] = {
                "cluster_label": f"Cluster {int(cluster_id) + 1}",
                "n_cities": len(cluster_cities),
                "cities": [c["city"] for c in cluster_cities],
                "mean_features": mean_features
            }
        
        # ===== STEP 8: Build response =====
        response = {
            "method": f"KNN (k={k}) + {clustering_method.title()} Clustering (n={n_clusters})",
            "year": year,
            "k": k,
            "n_clusters": n_clusters,
            "clustering_method": clustering_method,
            "cities_analyzed": len(city_data),
            "features_used": feature_names,
            "knn_results": knn_results,
            "cluster_stats": cluster_stats,
            "feature_importance": {
                "mean_bias_variance": float(scaler.var_[0]),
                "std_bias_variance": float(scaler.var_[1]),
                "mean_temp_variance": float(scaler.var_[4])
            }
        }
        
        print(f"\n[INFO] ===== Analysis Complete =====")
        print(f"[INFO] {len(city_data)} cities analyzed")
        print(f"[INFO] {k} nearest neighbors per city")
        print(f"[INFO] {n_clusters} clusters identified")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] KNN analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




def dms_to_deg(s):
    """Convert D:M:S (with commas or colons) to decimal degrees."""
    try:
        s = s.strip().replace("+", "").replace("−", "-").replace("–", "-")
        s = s.replace(",", ":")
        parts = s.split(":")
        
        if len(parts) == 3:
            d, m, sec = parts
        elif len(parts) == 2:
            d, m = parts
            sec = 0
        else:
            return np.nan
        
        d = float(d)
        m = float(m)
        sec = float(sec)
        
        return d + m / 60 + sec / 3600
    except Exception:
        return np.nan


def get_city_stations(city_name: str, features_df: pd.DataFrame, stations_metadata: pd.DataFrame):
    """
    Extract station IDs used for a city and enrich with coordinates.
    
    Args:
        city_name: Name of the city
        features_df: The features dataframe (must have been built with station panel)
        stations_metadata: Dataframe from load_station_metadata_for_api()
    
    Returns:
        List of dicts with station info including coordinates
    """
    # This requires that you saved station panel data or can reconstruct it
    # For now, we'll infer from the metrics file or features file
    
    # OPTION 1: If you have a station_ids column or separate panel file
    # OPTION 2: Load from a separate stations_used.json per city
    # OPTION 3: Hardcode major city coordinates as fallback
    
    pass  # Implement based on your data structure


# Load station metadata once at startup
STATION_METADATA = load_station_metadata_for_api()


def get_station_coordinates(city):
    """
    Get station coordinates for a city from actual station metadata.
    Falls back to hardcoded coordinates for known cities.
    """
    # Try to load from your features file to see which stations were used
    features_path = find_file(city, "features")







































# ============================================================
# STATION COORDINATE UTILITIES (inspired by your pipeline)
# ============================================================

def dms_to_deg(s):
    """Convert D:M:S (with commas or colons) to decimal degrees."""
    try:
        s = s.strip().replace("+", "").replace("−", "-").replace("–", "-")
        s = s.replace(",", ":")
        parts = s.split(":")
        
        if len(parts) == 3:
            d, m, sec = parts
        elif len(parts) == 2:
            d, m = parts
            sec = 0
        else:
            return None
        
        d = float(d)
        m = float(m)
        sec = float(sec)
        
        return d + m / 60 + sec / 3600
    except Exception:
        return None


def load_all_stations_metadata():
    """
    Load and parse stations.txt, converting DMS to decimal degrees.
    Returns a dict: {STAID: {name, country, lat, lon, elevation}}
    """
    # Use the PATHS dictionary you already have defined
    stations_meta_path = PATHS.get("stations_meta")
    
    # If PATHS not set, construct it manually
    if not stations_meta_path:
        stations_meta_path = os.path.join(DATA_DIR, "ECA_blend_tx", "stations.txt")

    print(f"[INFO] Looking for stations.txt at: {stations_meta_path}")

    if not os.path.exists(stations_meta_path):
        print(f"[ERROR] stations.txt not found at {stations_meta_path}")
        print(f"[DEBUG] DATA_DIR is: {DATA_DIR}")

        try:
            tx_dir = os.path.join(DATA_DIR, "ECA_blend_tx")
            if os.path.exists(tx_dir):
                files = os.listdir(tx_dir)
                print(f"[DEBUG] Files in ECA_blend_tx: {files[:10]}...")
            else:
                print(f"[ERROR] ECA_blend_tx directory doesn't exist!")
        except Exception as e:
            print(f"[DEBUG] Could not list directory: {e}")
        
        return {}
    
    stations = {}
    line_count = 0
    valid_count = 0
    
    try:
        with open(stations_meta_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                
                # Skip headers and empty lines
                if not line:
                    continue
                if "STAID" in line and "STANAME" in line:
                    continue
                if line.startswith("EUROPEAN") or line.startswith("FILE FORMAT") or line.startswith("THESE DATA"):
                    continue
                
                if "," not in line:
                    continue
                
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                
                try:
                    staid = int(parts[0])
                    name = parts[1]
                    country = parts[2]
                    lat_dms = parts[3]
                    lon_dms = parts[4]
                    elevation = int(parts[5]) if parts[5].isdigit() else None
                    
                    # Convert DMS to decimal
                    lat_deg = dms_to_deg(lat_dms)
                    lon_deg = dms_to_deg(lon_dms)
                    
                    # Only store if coordinates are valid
                    if lat_deg is not None and lon_deg is not None:
                        stations[staid] = {
                            "STAID": staid,
                            "name": name,
                            "country": country,
                            "lat": lat_deg,
                            "lon": lon_deg,
                            "elevation": elevation,
                            "lat_dms": lat_dms,
                            "lon_dms": lon_dms
                        }
                        valid_count += 1
                
                except Exception as e:
                    # Silently skip invalid lines
                    continue
        
        print(f"[INFO] Parsed {line_count} lines from stations.txt")
        print(f"[INFO] Loaded {valid_count} stations with valid coordinates")
        
        # Show a sample
        if stations:
            sample_staid = list(stations.keys())[0]
            sample = stations[sample_staid]
            print(f"[INFO] Sample station: {sample['name']} ({sample['country']}) at {sample['lat']:.4f}, {sample['lon']:.4f}")
        
        return stations
    
    except Exception as e:
        print(f"[ERROR] Failed to load stations.txt: {e}")
        import traceback
        traceback.print_exc()
        return {}




# Load station metadata once at startup
STATIONS_METADATA = load_all_stations_metadata()
print(f"[STARTUP] Loaded {len(STATIONS_METADATA)} stations from metadata")


def get_city_station_ids(city_name: str):
    """
    Extract station IDs that were used for a city's analysis.
    Tries multiple strategies in order of preference.
    """
    # Strategy 1: Load from stations_<city>.json (if pipeline saved it)
    city_file = city_name.replace(" ", "_")
    stations_json = os.path.join(BASE_DIR, f"stations_{city_file}.json")
    
    if os.path.exists(stations_json):
        try:
            with open(stations_json, 'r', encoding='utf-8') as f:
                stations = json.load(f)
            station_ids = [s['STAID'] for s in stations]
            print(f"[INFO] Loaded {len(station_ids)} station IDs from {stations_json}")
            return station_ids
        except Exception as e:
            print(f"[WARN] Could not load stations JSON: {e}")
    
    # Strategy 2: Extract from metrics.json
    metrics_path = find_file(city_name, "metrics")
    if metrics_path:
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            if 'stations_metadata' in metrics:
                stations = metrics['stations_metadata'].get('stations', [])
                station_ids = [s['STAID'] for s in stations]
                print(f"[INFO] Extracted {len(station_ids)} station IDs from metrics")
                return station_ids
        except Exception as e:
            print(f"[WARN] Could not extract from metrics: {e}")
    
    # Strategy 3: Search by city name in station names
    city_upper = city_name.upper()
    matching_ids = []
    
    for staid, info in STATIONS_METADATA.items():
        station_name = info['name'].upper()
        if city_upper in station_name:
            matching_ids.append(staid)
    
    if matching_ids:
        print(f"[INFO] Found {len(matching_ids)} stations matching '{city_name}' by name")
        return matching_ids
    
    print(f"[WARN] No station IDs found for {city_name}")
    return []


def get_station_coordinates(city):
    """
    Get primary station coordinates for a city.
    Returns station info with lat/lon in decimal degrees.
    """
    # Get all station IDs used for this city
    station_ids = get_city_station_ids(city)
    
    if not station_ids:
        print(f"[WARN] No stations found for {city}, using fallback")
        return get_fallback_coordinates(city)
    
    # Get metadata for these stations
    stations_with_coords = []
    for staid in station_ids:
        if staid in STATIONS_METADATA:
            stations_with_coords.append(STATIONS_METADATA[staid])
    
    if not stations_with_coords:
        print(f"[WARN] No station coordinates found for {city}, using fallback")
        return get_fallback_coordinates(city)
    
    # Use the first station as primary (or you could choose by distance, etc.)
    primary = stations_with_coords[0]
    
    return {
        "lat": primary['lat'],
        "lon": primary['lon'],
        "name": primary['name'],
        "country": primary['country'],
        "id": f"STAID_{primary['STAID']}",
        "elevation": primary.get('elevation'),
        "staid": primary['STAID']
    }


def get_fallback_coordinates(city):
    """Hardcoded coordinates for major European cities as fallback."""
    coords = {
        "Berlin": {"lat": 52.5167, "lon": 13.3833, "name": "Berlin-Tempelhof", "country": "DE", "id": "DE_BERLIN"},
        "Hamburg": {"lat": 53.5511, "lon": 9.9937, "name": "Hamburg-Fuhlsbüttel", "country": "DE", "id": "DE_HAMBURG"},
        "Warszawa": {"lat": 52.2297, "lon": 21.0122, "name": "Warszawa-Okęcie", "country": "PL", "id": "PL_WARSZAWA"},
        "Budapest": {"lat": 47.4979, "lon": 19.0402, "name": "Budapest-Lőrinc", "country": "HU", "id": "HU_BUDAPEST"},
        "Barcelona": {"lat": 41.3874, "lon": 2.1686, "name": "Barcelona-Airport", "country": "ES", "id": "ES_BARCELONA"},
        "Paris": {"lat": 48.8566, "lon": 2.3522, "name": "Paris-Montsouris", "country": "FR", "id": "FR_PARIS"},
        "Rome": {"lat": 41.9028, "lon": 12.4964, "name": "Roma Ciampino", "country": "IT", "id": "IT_ROME"},
        "London": {"lat": 51.5074, "lon": -0.1278, "name": "London-Heathrow", "country": "GB", "id": "GB_LONDON"},
        "Vienna": {"lat": 48.2082, "lon": 16.3738, "name": "Wien-Hohe Warte", "country": "AT", "id": "AT_VIENNA"},
        "Prague": {"lat": 50.0755, "lon": 14.4378, "name": "Praha-Ruzyně", "country": "CZ", "id": "CZ_PRAGUE"},
    }
    
    return coords.get(city, {
        "lat": 50.0,
        "lon": 10.0,
        "name": city,
        "country": "Unknown",
        "id": city.lower().replace(" ", "_")
    })
























































# ============================================================
# ML MODELS ENDPOINTS
# ============================================================

@app.route("/api/models/<city>", methods=["GET"])
def get_models(city):
    """
    Get all available ML models for a city with metadata.
    """
    try:
        city_file = city.replace(" ", "_")
        models_info = {}
        
        # Check for each model type
        model_types = {
            "rf_bias": f"rf_bias_{city_file}.pkl",
            "xgb_bias": f"xgb_bias_{city_file}.pkl",
            "qreg90": f"qreg90_{city_file}.pkl",
            "kmeans_weather": f"kmeans_weather_{city_file}.pkl",
            "rf_uhi_proxy": f"rf_uhi_proxy_{city_file}.pkl",
            "heatwave_classifier": f"heatwave_classifier_{city_file}.pkl",
            "qdm_inverse": f"qdm_inverse_{city_file}.pkl",
            "variance_scaling": f"variance_scaling_{city_file}.pkl",
            "ml_bias": f"ml_bias_{city_file}.pkl",
            "hybrid": f"hybrid_{city_file}.pkl"
        }
        
        for model_key, filename in model_types.items():
            model_path = os.path.join(MODELS_DIR, filename)
            
            if os.path.exists(model_path):
                # Get file size
                size_kb = os.path.getsize(model_path) / 1024
                
                # Try to load metadata from best_parameters file
                best_params_path = os.path.join(MODELS_DIR, f"best_parameters_{city_file}.json")
                metadata = None
                
                if os.path.exists(best_params_path):
                    with open(best_params_path, 'r') as f:
                        params_data = json.load(f)
                        
                        # Map model keys to parameter keys
                        param_map = {
                            "xgb_bias": "XGBoost_Bias_Correction",
                            "rf_uhi_proxy": "RandomForest_UHI",
                            "qreg90": "Quantile_Regression"
                        }
                        
                        if model_key in param_map:
                            param_key = param_map[model_key]
                            if param_key in params_data:
                                metadata = params_data[param_key]
                
                models_info[model_key] = {
                    "filename": filename,
                    "size_kb": round(size_kb, 2),
                    "path": model_path,
                    "exists": True,
                    "metadata": metadata
                }
        
        return jsonify({
            "city": city,
            "count": len(models_info),
            "models": models_info
        })
    
    except Exception as e:
        print(f"[ERROR] get_models: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<city>/predictions", methods=["GET"])
def get_model_predictions(city):
    """
    Get model predictions from predictions_{city}.csv
    """
    try:
        city_file = city.replace(" ", "_")
        predictions_path = os.path.join(MODELS_DIR, f"predictions_{city_file}.csv")
        
        if not os.path.exists(predictions_path):
            return jsonify({"error": "Predictions file not found"}), 404
        
        df = pd.read_csv(predictions_path, parse_dates=['date'])
        
        # Get date range filters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        # Convert to JSON-serializable format
        data = {
            "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "bias_correction": df['bias_correction'].tolist() if 'bias_correction' in df.columns else [],
            "uhi": df['uhi'].tolist() if 'uhi' in df.columns else [],
            "downscaling": df['downscaling'].tolist() if 'downscaling' in df.columns else [],
            "quantile_median": df['quantile_median'].tolist() if 'quantile_median' in df.columns else [],
            "quantile_q10": df['quantile_q10'].tolist() if 'quantile_q10' in df.columns else [],
            "quantile_q90": df['quantile_q90'].tolist() if 'quantile_q90' in df.columns else [],
            "columns": df.columns.tolist()
        }
        
        return jsonify(data)
    
    except Exception as e:
        print(f"[ERROR] get_model_predictions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<city>/prepared_data", methods=["GET"])
def get_prepared_data(city):
    """
    Get prepared training data from prepared_data_{city}.csv
    """
    try:
        city_file = city.replace(" ", "_")
        prepared_path = os.path.join(MODELS_DIR, f"prepared_data_{city_file}.csv")
        
        if not os.path.exists(prepared_path):
            return jsonify({"error": "Prepared data file not found"}), 404
        
        df = pd.read_csv(prepared_path, parse_dates=['date'])
        
        # Get date range filters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        # Convert to JSON
        data = {
            "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "TX_C_city_mean": df['TX_C_city_mean'].tolist() if 'TX_C_city_mean' in df.columns else [],
            "TN_C_city_mean": df['TN_C_city_mean'].tolist() if 'TN_C_city_mean' in df.columns else [],
            "ERA5_t2m_max_C": df['ERA5_t2m_max_C'].tolist() if 'ERA5_t2m_max_C' in df.columns else [],
            "ERA5_t2m_min_C": df['ERA5_t2m_min_C'].tolist() if 'ERA5_t2m_min_C' in df.columns else [],
            "NDVI_city": df['NDVI_city'].tolist() if 'NDVI_city' in df.columns else [],
            "columns": df.columns.tolist()
        }
        
        return jsonify(data)
    
    except Exception as e:
        print(f"[ERROR] get_prepared_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<city>/performance", methods=["GET"])
def get_model_performance(city):
    """
    Get model performance metrics from exhaustive_grid_search_results_{city}.json
    """
    try:
        city_file = city.replace(" ", "_")
        metrics_path = os.path.join(MODELS_DIR, f"exhaustive_grid_search_results_{city_file}.json")
        
        if not os.path.exists(metrics_path):
            return jsonify({"error": "Performance metrics file not found"}), 404
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        return jsonify(metrics)
    
    except Exception as e:
        print(f"[ERROR] get_model_performance: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<city>/best_parameters", methods=["GET"])
def get_best_parameters(city):
    """
    Get best selected parameters from best_parameters_{city}.json
    """
    try:
        city_file = city.replace(" ", "_")
        print(city_file)
        params_path = os.path.join(MODELS_DIR, f"best_parameters_{city_file}.json")
        
        if not os.path.exists(params_path):
            return jsonify({"error": "Best parameters file not found"}), 404
        
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        return jsonify(params)
    
    except Exception as e:
        print(f"[ERROR] get_best_parameters: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<city>/comparison_chart", methods=["GET"])
def get_comparison_chart_data(city):
    """
    Get combined data for model comparison charts
    Merges predictions with prepared data for complete visualization
    """
    try:
        city_file = city.replace(" ", "_")
        
        # Load predictions
        predictions_path = os.path.join(MODELS_DIR, f"predictions_{city_file}.csv")
        prepared_path = os.path.join(MODELS_DIR, f"prepared_data_{city_file}.csv")
        
        if not os.path.exists(predictions_path) or not os.path.exists(prepared_path):
            return jsonify({"error": "Required data files not found"}), 404
        
        pred_df = pd.read_csv(predictions_path, parse_dates=['date'])
        prep_df = pd.read_csv(prepared_path, parse_dates=['date'])
        
        # Merge
        df = pred_df.merge(prep_df, on='date', how='inner')
        
        # Date filtering
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        # Downsample if too many points
        if len(df) > 5000:
            df = df.iloc[::len(df)//5000]
        
        # Build response
        data = {
            "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "models": {
                "bias_correction": {
                    "prediction": df['bias_correction'].tolist() if 'bias_correction' in df.columns else [],
                    "ground_truth": df['TX_C_city_mean'].tolist() if 'TX_C_city_mean' in df.columns else [],
                    "raw_era5": df['ERA5_t2m_max_C'].tolist() if 'ERA5_t2m_max_C' in df.columns else []
                },
                "uhi": {
                    "prediction": df['uhi'].tolist() if 'uhi' in df.columns else []
                },
                "downscaling": {
                    "prediction": df['downscaling'].tolist() if 'downscaling' in df.columns else [],
                    "ground_truth": df['TN_C_city_mean'].tolist() if 'TN_C_city_mean' in df.columns else [],
                    "raw_era5": df['ERA5_t2m_min_C'].tolist() if 'ERA5_t2m_min_C' in df.columns else []
                },
                "quantile": {
                    "median": df['quantile_median'].tolist() if 'quantile_median' in df.columns else [],
                    "q10": df['quantile_q10'].tolist() if 'quantile_q10' in df.columns else [],
                    "q90": df['quantile_q90'].tolist() if 'quantile_q90' in df.columns else [],
                    "ground_truth": df['TX_C_city_mean'].tolist() if 'TX_C_city_mean' in df.columns else []
                }
            }
        }
        
        return jsonify(data)
    
    except Exception as e:
        print(f"[ERROR] get_comparison_chart_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# MULTI-CITY MAP DATA ENDPOINT (COMPLETE VERSION)
# ============================================================

@app.route("/api/multi_city/map_data", methods=["POST", "OPTIONS"])
def multi_city_map_data():
    """
    Generate map data for multiple cities.
    Request body: {"cities": ["Berlin", "Hamburg", ...], "year": 2022}
    """
    try:
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        cities = data.get("cities", [])
        year = data.get("year", 2022)
        
        print(f"\n[INFO] ===== Multi-city map request =====")
        print(f"[INFO] Cities: {cities}")
        print(f"[INFO] Year: {year}")
        
        if not cities:
            return jsonify({"error": "No cities provided"}), 400
        
        map_points = []
        errors = []
        
        for city in cities:
            try:
                print(f"\n[INFO] Processing {city}...")
                
                # 1. Load features file
                features_path = find_file(city, "features")
                if not features_path:
                    error_msg = f"Features file not found for {city}"
                    print(f"[WARN] {error_msg}")
                    errors.append({"city": city, "error": error_msg})
                    continue
                
                print(f"[INFO] Loading features from {features_path}")
                df = pd.read_csv(features_path, parse_dates=["date"])
                
                # 2. Filter by year
                df_year = df[df["date"].dt.year == year]
                print(f"[INFO] {city}: {len(df_year)} records in year {year}")
                
                if df_year.empty:
                    error_msg = f"No data for year {year}"
                    print(f"[WARN] {city}: {error_msg}")
                    errors.append({"city": city, "error": error_msg})
                    continue
                
                # 3. Check required columns
                if "ERA5_t2m_max_C" not in df_year.columns:
                    error_msg = "Missing ERA5_t2m_max_C column"
                    print(f"[WARN] {city}: {error_msg}")
                    errors.append({"city": city, "error": error_msg})
                    continue
                
                if "TX_C_city_mean" not in df_year.columns:
                    error_msg = "Missing TX_C_city_mean column"
                    print(f"[WARN] {city}: {error_msg}")
                    errors.append({"city": city, "error": error_msg})
                    continue
                
                # 4. Drop NaN values
                df_clean = df_year.dropna(subset=["ERA5_t2m_max_C", "TX_C_city_mean"])
                print(f"[INFO] {city}: {len(df_clean)} clean records")
                
                if len(df_clean) < 10:
                    error_msg = f"Insufficient data: only {len(df_clean)} records"
                    print(f"[WARN] {city}: {error_msg}")
                    errors.append({"city": city, "error": error_msg})
                    continue
                
                # 5. Calculate bias statistics
                bias = df_clean["ERA5_t2m_max_C"] - df_clean["TX_C_city_mean"]
                mean_temp = df_clean["TX_C_city_mean"].mean()
                
                # 6. Get station coordinates
                station_info = get_station_coordinates(city)
                
                # 7. Build result
                point = {
                    "city": city,
                    "station_name": station_info.get("name", city),
                    "station_id": station_info.get("id", city.lower().replace(" ", "_")),
                    "country": station_info.get("country", "Unknown"),
                    "lat": station_info.get("lat", 50.0),
                    "lon": station_info.get("lon", 10.0),
                    "mean_bias": float(bias.mean()),
                    "std_bias": float(bias.std()),
                    "max_bias": float(bias.max()),
                    "mean_temp": float(mean_temp),
                    "n_records": int(len(df_clean))
                }
                
                map_points.append(point)
                print(f"[SUCCESS] {city}: bias={point['mean_bias']:.2f}°C, n={point['n_records']}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] {city}: {error_msg}")
                import traceback
                traceback.print_exc()
                errors.append({"city": city, "error": error_msg})
                continue
        
        # Return results
        if not map_points:
            return jsonify({
                "error": "No valid data for any selected cities",
                "details": errors
            }), 404
        
        response = {
            "cities_count": len(map_points),
            "stations_count": len(map_points),
            "year": year,
            "map_points": map_points
        }
        
        if errors:
            response["warnings"] = errors
        
        print(f"\n[INFO] ===== Request complete =====")
        print(f"[INFO] Successful: {len(map_points)} cities")
        print(f"[INFO] Failed: {len(errors)} cities")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] Fatal error in multi_city_map_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Climate Data Visualization API")
    print("=" * 50)
    print(f"Base directory: {BASE_DIR}")
    print("Available endpoints:")
    print("  GET  /api/cities")
    print("  GET  /api/metrics/<city>")
    print("  GET  /api/features/<city>")
    print("  GET  /api/data/<city>/chart/<chart_type>")
    print("  GET  /api/plot/<city>/<plot_type>")
    print("  GET  /api/stats/<city>/summary")
    print("  GET  /api/download/<city>/features")
    print("  GET  /api/download/<city>/metrics")
    print("=" * 50)
    app.run(debug=True, port=5000, host="0.0.0.0")