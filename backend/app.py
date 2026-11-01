from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image, UnidentifiedImageError
import os
from database import get_db_connection
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from werkzeug.utils import secure_filename
from uuid import uuid4
from disease_predictor import predict_disease
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from pathlib import Path
import csv
from werkzeug.security import generate_password_hash, check_password_hash

from crop_recommender import recommend_crops

app = Flask(__name__)
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect(app)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[]
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)


app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = 1800
UPLOAD_FOLDER = Path(app.root_path) / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def is_valid_image(file):
    try:
        file.stream.seek(0)

        with Image.open(file.stream) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                return False

            image.verify()

        file.stream.seek(0)
        return True

    except (UnidentifiedImageError, OSError, ValueError):
        file.stream.seek(0)
        return False


app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "crop_production_cleaned.csv"


@app.route("/")
def home():

    if "farmer_id" not in session:
        return redirect(url_for("login"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT farmer_id, farmer_name, phone, state, district
            FROM farmers
            WHERE farmer_id = %s
        """, (session["farmer_id"],))

        farmer = cursor.fetchone()

        if not farmer:
            session.clear()
            return redirect(url_for("login"))

        session["farmer_name"] = farmer["farmer_name"]

        return render_template(
            "index.html",
            farmer_name=farmer["farmer_name"],
            farmer_phone=farmer["phone"],
            farmer_state=farmer["state"],
            farmer_district=farmer["district"]
        )


    except Exception as error:
        print("Dashboard Error:", error)
        return "Unable to load dashboard.", 500

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()
@app.route("/edit-profile", methods=["GET", "POST"])
def edit_profile():

    if "farmer_id" not in session:
        return redirect(url_for("login"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        farmer_id = session["farmer_id"]

        if request.method == "POST":
            farmer_name = request.form.get("farmer_name", "").strip()
            phone = request.form.get("phone", "").strip()
            state = request.form.get("state", "").strip()
            district = request.form.get("district", "").strip()

            if not all([farmer_name, phone, state, district]):
                flash("Please fill all profile fields.")
                return redirect(url_for("edit_profile"))

            if not phone.isdigit() or len(phone) != 10:
                flash("Please enter a valid 10-digit phone number.")
                return redirect(url_for("edit_profile"))

            cursor.execute("""
                UPDATE farmers
                SET farmer_name = %s,
                    phone = %s,
                    state = %s,
                    district = %s
                WHERE farmer_id = %s
            """, (farmer_name, phone, state, district, farmer_id))

            connection.commit()

            session["farmer_name"] = farmer_name

            flash("Profile updated successfully!")
            return redirect(url_for("home"))

        cursor.execute("""
            SELECT farmer_name, phone, state, district
            FROM farmers
            WHERE farmer_id = %s
        """, (farmer_id,))

        farmer = cursor.fetchone()

        if not farmer:
            session.clear()
            return redirect(url_for("login"))

        return render_template("edit_profile.html", farmer=farmer)

    except Exception as error:
        if connection:
            connection.rollback()

        print("Edit Profile Error:", error)
        flash("Unable to update profile. Phone number may already be registered.")
        return redirect(url_for("home"))

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()
# ==========================================
# DISTRICT API
# ==========================================


@app.route("/api/districts")
def get_districts():

    # Allow only logged-in farmers
    if "farmer_id" not in session:
        return jsonify({"error": "Please login first."}), 401

    selected_state = request.args.get("state", "").strip()

    if not selected_state:
        return jsonify([])

    districts = set()

    try:
        with open(
            DATASET_PATH,
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                state = (row.get("State_Name") or "").strip()
                district = (row.get("District_Name") or "").strip()

                if state.casefold() == selected_state.casefold() and district:
                    districts.add(district)

        return jsonify(sorted(districts))

    except FileNotFoundError:
        return jsonify({"error": "Dataset not found"}), 500

# ==========================================
# CROP ADVISOR
# ==========================================


@app.route("/advisor", methods=["GET", "POST"])
def advisor():

    # Login protection
    if "farmer_id" not in session:
        return redirect(url_for("login"))

    submitted = False
    farm_data = {}
    recommendation_result = None

    if request.method == "POST":

        farm_data = {
            "state": request.form.get("state", "").strip(),
            "district": request.form.get("district", "").strip(),
            "soil_type": request.form.get("soil_type", ""),
            "ph": request.form.get("ph", ""),
            "nitrogen": request.form.get("nitrogen", ""),
            "phosphorus": request.form.get("phosphorus", ""),
            "potassium": request.form.get("potassium", ""),
            "water": request.form.get("water", ""),
            "season": request.form.get("season", "")
        }

        submitted = True

        # Validate state and district
        try:
            with open(
                DATASET_PATH,
                mode="r",
                encoding="utf-8-sig",
                newline=""
            ) as file:

                reader = csv.DictReader(file)

                valid_location = any(
                    (row.get("State_Name") or "").strip().casefold()
                    == farm_data["state"].casefold()
                    and
                    (row.get("District_Name") or "").strip().casefold()
                    == farm_data["district"].casefold()
                    for row in reader
                )

        except FileNotFoundError:
            valid_location = False

        if not valid_location:
            recommendation_result = {
                "success": False,
                "message": "Please select a valid state and district."
            }

        else:
            recommendation_result = recommend_crops(
    state=farm_data["state"],
    district=farm_data["district"],
    season=farm_data["season"],
    soil_type=farm_data["soil_type"],
    ph=farm_data["ph"],
    water=farm_data["water"],
    nitrogen=farm_data["nitrogen"],
    phosphorus=farm_data["phosphorus"],
    potassium=farm_data["potassium"]
)


    return render_template(
        "advisor.html",
        submitted=submitted,
        farm_data=farm_data,
        recommendation_result=recommendation_result
    )


@app.route("/crop-doctor", methods=["GET", "POST"])
def crop_doctor():

    if "farmer_id" not in session:
        return redirect(url_for("login"))


    result = None

    if request.method == "POST":

        if "leaf_image" not in request.files:
            result = {
                "success": False,
                "message": "Please select a leaf image."
            }

        else:
            file = request.files["leaf_image"]

            if file.filename == "":
                result = {
                    "success": False,
                    "message": "No image selected."
                }

            elif "." not in file.filename or file.filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
                result = {
                    "success": False,
                    "message": "Please upload JPG, JPEG, PNG or WEBP."
                }

            elif not is_valid_image(file):
                result = {
                    "success": False,
                    "message": "Invalid or corrupted image. Please upload a genuine JPG, PNG or WEBP image."
                }

            else:
                extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
                filename = f"{uuid4().hex}.{extension}"
                image_path = UPLOAD_FOLDER / filename

                try:
                    file.save(image_path)
                    result = predict_disease(image_path)

                finally:
                    if image_path.exists():
                        image_path.unlink()

    return render_template(
        "crop_doctor.html",
        result=result
    )



@app.route("/crop-planner", methods=["GET", "POST"])
def crop_planner():

    if "farmer_id" not in session:
        return redirect(url_for("login"))

    result = None

    if request.method == "POST":

        connection = None
        cursor = None

        try:
            import math

            crop = request.form.get("crop", "").strip()
            area = float(request.form.get("area", 0))
            season = request.form.get("season", "").strip()
            irrigation = request.form.get("irrigation", "").strip()

            expected_yield = float(request.form.get("expected_yield", 0))
            selling_price = float(request.form.get("selling_price", 0))

            # Six farming expense categories
            seed_cost = float(request.form.get("seed_cost", 0))
            fertilizer_cost = float(request.form.get("fertilizer_cost", 0))
            labour_cost = float(request.form.get("labour_cost", 0))
            irrigation_cost = float(request.form.get("irrigation_cost", 0))
            pesticide_cost = float(request.form.get("pesticide_cost", 0))
            other_cost = float(request.form.get("other_cost", 0))

            allowed_crops = {
                "Bajra", "Jowar", "Gram",
                "Moong(Green Gram)", "Moth",
                "Arhar/Tur", "Urad"
            }

            if crop not in allowed_crops:
                raise ValueError("Please select a valid crop.")

            if season not in {"Kharif", "Rabi", "Summer"}:
                raise ValueError("Please select a valid season.")

            if irrigation not in {"Rainfed", "Limited", "Adequate"}:
                raise ValueError("Please select valid water availability.")

            numeric_values = [
                area, expected_yield, selling_price,
                seed_cost, fertilizer_cost, labour_cost,
                irrigation_cost, pesticide_cost, other_cost
            ]

            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError("Please enter valid numeric values.")

            if not 0.1 <= area <= 10000:
                raise ValueError("Land area must be between 0.1 and 10,000 acres.")

            if not 0.1 <= expected_yield <= 1000:
                raise ValueError("Expected yield must be between 0.1 and 1,000 quintals per acre.")

            if not 1 <= selling_price <= 1000000:
                raise ValueError("Please enter a valid selling price.")

            expenses = [
                seed_cost, fertilizer_cost, labour_cost,
                irrigation_cost, pesticide_cost, other_cost
            ]

            if any(cost < 0 or cost > 10000000 for cost in expenses):
                raise ValueError("Each expense must be between ₹0 and ₹10,000,000 per acre.")

            # Calculate expenses
            total_cost_per_acre = sum(expenses)
            total_cost = total_cost_per_acre * area

            # Production and revenue
            total_production = area * expected_yield
            total_revenue = total_production * selling_price

            # Profit calculations
            net_profit = total_revenue - total_cost
            profit_per_acre = net_profit / area

            break_even_price = total_cost_per_acre / expected_yield

            # Connect to MySQL
            connection = get_db_connection()
            cursor = connection.cursor()

            # Save crop plan
            plan_query = """
                INSERT INTO crop_plans (
                    farmer_id, crop, area, season, irrigation,
                    expected_yield, selling_price,
                    total_production, total_revenue,
                    total_cost, net_profit
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            plan_values = (
                session["farmer_id"],
                crop,
                area,
                season,
                irrigation,
                expected_yield,
                selling_price,
                total_production,
                total_revenue,
                total_cost,
                net_profit
            )

            cursor.execute(plan_query, plan_values)

            # Get newly created plan ID
            plan_id = cursor.lastrowid

            # Save six expense categories
            expense_query = """
                INSERT INTO crop_expenses (
                    plan_id, seed_cost, fertilizer_cost,
                    labour_cost, irrigation_cost,
                    pesticide_cost, other_cost
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            expense_values = (
                plan_id,
                seed_cost,
                fertilizer_cost,
                labour_cost,
                irrigation_cost,
                pesticide_cost,
                other_cost
            )

            cursor.execute(expense_query, expense_values)

            # Commit both records together
            connection.commit()

            # Prepare results for webpage
            result = {
                "success": True,
                "plan_id": plan_id,
                "saved": True,

                "crop": crop,
                "area": area,
                "season": season,
                "irrigation": irrigation,

                "total_production": round(total_production, 2),
                "total_revenue": round(total_revenue, 2),
                "total_cost": round(total_cost, 2),
                "total_cost_per_acre": round(total_cost_per_acre, 2),

                "net_profit": round(net_profit, 2),
                "profit_per_acre": round(profit_per_acre, 2),
                "break_even_price": round(break_even_price, 2),

                "selling_price": selling_price,
                "expected_yield": expected_yield,

                "seed_cost": seed_cost,
                "fertilizer_cost": fertilizer_cost,
                "labour_cost": labour_cost,
                "irrigation_cost": irrigation_cost,
                "pesticide_cost": pesticide_cost,
                "other_cost": other_cost
            }

        except (ValueError, TypeError) as error:

            if connection:
                connection.rollback()

            result = {
                "success": False,
                "message": str(error)
            }

        except Exception as error:

            if connection:
                connection.rollback()

            print("Crop Planner Database Error:", error)

            result = {
                "success": False,
                "message": "Unable to save your crop plan. Please check the database connection."
            }

        finally:

            if cursor:
                cursor.close()

            if connection and connection.is_connected():
                connection.close()

    return render_template("planner.html", result=result)

@app.route("/crop-history")
def crop_history():

    if "farmer_id" not in session:
        return redirect(url_for("login"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                plan_id,
                crop,
                area,
                season,
                irrigation,
                expected_yield,
                selling_price,
                total_production,
                total_revenue,
                total_cost,
                net_profit,
                created_at
            FROM crop_plans
            WHERE farmer_id = %s
            ORDER BY created_at DESC
        """, (session["farmer_id"],))

        plans = cursor.fetchall()

        return render_template(
            "crop_history.html",
            plans=plans
        )

    except Exception as error:
        print("Crop History Error:", error)
        return "Unable to load Crop Plan History.", 500

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

@app.route("/api/weather")
def get_weather():

    city = request.args.get("city", "").strip()

    if not city:
        return jsonify({
            "success": False,
            "message": "Please enter a city or district name."
        }), 400

    try:
        # STEP 1: Find the location coordinates

        geo_params = urlencode({
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        })

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + geo_params
        )

        with urlopen(geo_url, timeout=45) as response:
            geo_data = json.loads(response.read().decode("utf-8"))

        locations = geo_data.get("results", [])

        if not locations:
            return jsonify({
                "success": False,
                "message": "Location not found. Try another city or district."
            }), 404

        location = locations[0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # STEP 2: Get weather forecast

        weather_params = urlencode({
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "precipitation,"
                "rain,"
                "weather_code,"
                "wind_speed_10m"
            ),
            "daily": (
                "weather_code,"
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max,"
                "precipitation_sum"
            ),
            "forecast_days": 7,
            "timezone": "auto"
        })

        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + weather_params
        )

        with urlopen(weather_url, timeout=45) as response:
            weather_data = json.loads(
                response.read().decode("utf-8")
            )

        # STEP 3: Prepare response

        return jsonify({
            "success": True,

            "location": {
                "name": location.get("name"),
                "admin1": location.get("admin1", ""),
                "country": location.get("country", "")
            },

            "current": weather_data.get("current", {}),

            "daily": weather_data.get("daily", {}),

            "units": {
                "temperature": "°C",
                "humidity": "%",
                "wind_speed": "km/h",
                "rainfall": "mm"
            }
        })

    except Exception as error:

        print("Weather API Error:", error)

        return jsonify({
            "success": False,
            "message": "Unable to fetch weather data. Please try again."
        }), 500

@app.route("/weather")
def weather_page():
    return render_template("weather.html")

@app.route("/farm-analytics")
def farm_analytics():
    if "farmer_id" not in session:
        return redirect(url_for("login"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        farmer_id = session["farmer_id"]

        # Statistics for the currently logged-in farmer only
        cursor.execute("""
            SELECT
                COUNT(*) AS total_plans,
                COALESCE(SUM(area), 0) AS total_area,
                COALESCE(SUM(total_production), 0) AS total_production,
                COALESCE(SUM(total_revenue), 0) AS total_revenue,
                COALESCE(SUM(total_cost), 0) AS total_expenses,
                COALESCE(SUM(net_profit), 0) AS total_profit
            FROM crop_plans
            WHERE farmer_id = %s
        """, (farmer_id,))

        stats = cursor.fetchone()

        # Crop-wise performance for the currently logged-in farmer only
        cursor.execute("""
            SELECT
                crop,
                COUNT(*) AS plans,
                SUM(area) AS area,
                SUM(total_production) AS production,
                SUM(total_revenue) AS revenue,
                SUM(total_cost) AS expenses,
                SUM(net_profit) AS profit
            FROM crop_plans
            WHERE farmer_id = %s
            GROUP BY crop
            ORDER BY profit DESC
        """, (farmer_id,))

        crop_data = cursor.fetchall()

        return render_template(
            "analytics.html",
            stats=stats,
            crop_data=crop_data
        )

    except Exception as error:
        print("Farm Analytics Error:", error)
        return "Unable to load Farm Analytics. Check the database connection.", 500

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ==============================
# FARMER REGISTRATION
# ==============================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        farmer_name = request.form.get("farmer_name", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        state = request.form.get("state", "").strip()
        district = request.form.get("district", "").strip()

        if not all([farmer_name, phone, password, state, district]):
            flash("Please fill in all fields.")
            return redirect(url_for("register"))

        if not phone.isdigit() or len(phone) != 10:
            flash("Enter a valid 10-digit phone number.")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must contain at least 8 characters.")
            return redirect(url_for("register"))

        connection = None
        cursor = None

        try:
            connection = get_db_connection()
            cursor = connection.cursor()

            password_hash = generate_password_hash(password)

            cursor.execute("""
                INSERT INTO farmers
                (farmer_name, phone, state, district, password_hash)
                VALUES (%s, %s, %s, %s, %s)
            """, (farmer_name, phone, state, district, password_hash))

            connection.commit()

            flash("Registration successful! Please login.")
            return redirect(url_for("login"))

        except Exception as error:
            if connection:
                connection.rollback()

            print("Registration Error:", error)
            flash("Registration failed. This phone number may already be registered.")

        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

        return redirect(url_for("register"))

    return render_template("register.html")


# ==============================
# FARMER LOGIN
# ==============================

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def login():

    if request.method == "POST":

        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")

        connection = None
        cursor = None

        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)

            cursor.execute("""
                SELECT farmer_id, farmer_name, password_hash
                FROM farmers
                WHERE phone = %s
            """, (phone,))

            farmer = cursor.fetchone()

            if farmer and farmer["password_hash"] and check_password_hash(
                farmer["password_hash"], password
            ):

                session.clear()
                session.permanent = True
                session["farmer_id"] = farmer["farmer_id"]
                session["farmer_name"] = farmer["farmer_name"]

                return redirect("/")

            flash("Invalid phone number or password.")

        except Exception as error:
            print("Login Error:", error)
            flash("Unable to login. Please try again.")

        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

        return redirect(url_for("login"))

    return render_template("login.html")


# ==============================
# FARMER LOGOUT
# ==============================

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been logged out successfully.")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=False)
