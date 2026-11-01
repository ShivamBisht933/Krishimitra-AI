
import pandas as pd
from pathlib import Path


# ==========================================
# KRISHIMITRA AI - CROP RECOMMENDATION ENGINE
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "crop_production_cleaned.csv"


TARGET_CROPS = {
    "Bajra": "Pearl Millet (Bajra)",
    "Jowar": "Sorghum (Jowar)",
    "Gram": "Chickpea (Gram)",
    "Moong(Green Gram)": "Green Gram (Moong)",
    "Moth": "Moth Bean",
    "Arhar/Tur": "Pigeon Pea (Arhar)",
    "Urad": "Black Gram (Urad)"
}


# Prototype screening rules.
# These rules require validation by regional agricultural experts.

CROP_PROFILES = {
    "Bajra": {
        "seasons": ["Kharif"],
        "soils": ["Sandy", "Sandy Loam", "Loamy"],
        "water": ["Low", "Medium"],
        "ph_min": 5.5,
        "ph_max": 8.5
    },

    "Jowar": {
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Loamy", "Clay", "Black Soil", "Sandy Loam"],
        "water": ["Low", "Medium", "High"],
        "ph_min": 5.5,
        "ph_max": 8.5
    },

    "Gram": {
        "seasons": ["Rabi"],
        "soils": ["Loamy", "Sandy Loam", "Black Soil"],
        "water": ["Low", "Medium"],
        "ph_min": 6.0,
        "ph_max": 8.5
    },

    "Moong(Green Gram)": {
        "seasons": ["Kharif", "Zaid"],
        "soils": ["Sandy Loam", "Loamy", "Black Soil"],
        "water": ["Low", "Medium"],
        "ph_min": 6.0,
        "ph_max": 8.0
    },

    "Moth": {
        "seasons": ["Kharif"],
        "soils": ["Sandy", "Sandy Loam", "Loamy"],
        "water": ["Low"],
        "ph_min": 6.0,
        "ph_max": 8.5
    },

    "Arhar/Tur": {
        "seasons": ["Kharif"],
        "soils": ["Loamy", "Sandy Loam", "Black Soil"],
        "water": ["Medium"],
        "ph_min": 6.0,
        "ph_max": 7.5
    },

    "Urad": {
        "seasons": ["Kharif"],
        "soils": ["Loamy", "Sandy Loam", "Clay"],
        "water": ["Medium", "High"],
        "ph_min": 5.5,
        "ph_max": 7.5
    }
}


def parse_optional_number(value, field_name):

    if value is None or str(value).strip() == "":
        return None

    try:
        number = float(value)

        if not pd.notna(number) or number < 0:
            raise ValueError

        return number

    except (ValueError, TypeError):
        raise ValueError(
            f"Please enter a valid non-negative {field_name} value."
        )


def recommend_crops(
    state,
    district,
    season,
    soil_type="",
    ph=None,
    water="",
    nitrogen=None,
    phosphorus=None,
    potassium=None
):

    # ======================================
    # 1. LOAD DATASET
    # ======================================

    try:
        data = pd.read_csv(DATASET_PATH)

    except FileNotFoundError:
        return {
            "success": False,
            "message": "Crop dataset could not be found."
        }

    except Exception as error:
        print("Crop Dataset Error:", error)

        return {
            "success": False,
            "message": "Unable to read the crop dataset."
        }

    required_columns = {
        "State_Name",
        "District_Name",
        "Season",
        "Crop",
        "Area",
        "Production",
        "Crop_Year"
    }

    if not required_columns.issubset(data.columns):
        return {
            "success": False,
            "message": "Crop dataset is missing required columns."
        }

    # Clean text columns

    text_columns = [
        "State_Name",
        "District_Name",
        "Season",
        "Crop"
    ]

    for column in text_columns:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Convert numeric columns

    data["Area"] = pd.to_numeric(
        data["Area"], errors="coerce"
    )

    data["Production"] = pd.to_numeric(
        data["Production"], errors="coerce"
    )

    data["Crop_Year"] = pd.to_numeric(
        data["Crop_Year"], errors="coerce"
    )

    # ======================================
    # 2. FILTER LOCATION AND SEASON
    # ======================================

    state = str(state or "").strip()
    district = str(district or "").strip()
    season = str(season or "").strip()

    filtered = data[
        (data["State_Name"].str.casefold() == state.casefold()) &
        (data["District_Name"].str.casefold() == district.casefold()) &
        (data["Season"].str.casefold() == season.casefold())
    ]

    if filtered.empty:
        return {
            "success": False,
            "message": (
                "No historical records found for this "
                "district and season."
            )
        }

    # Only target millet and pulse crops

    filtered = filtered[
        filtered["Crop"].isin(TARGET_CROPS.keys())
    ]

    # Keep records with valid area and production

    filtered = filtered[
        (filtered["Area"] > 0) &
        (filtered["Production"] >= 0)
    ].copy()

    if filtered.empty:
        return {
            "success": False,
            "message": (
                "No usable millet or pulse production "
                "records found for this location and season."
            )
        }

    # ======================================
    # 3. PREPARE FARM INPUTS
    # ======================================

    soil_type = str(soil_type or "").strip()
    water = str(water or "").strip()

    try:
        ph_value = parse_optional_number(ph, "pH")

        nitrogen_value = parse_optional_number(
            nitrogen, "Nitrogen"
        )

        phosphorus_value = parse_optional_number(
            phosphorus, "Phosphorus"
        )

        potassium_value = parse_optional_number(
            potassium, "Potassium"
        )

    except ValueError as error:
        return {
            "success": False,
            "message": str(error)
        }

    if ph_value is not None and ph_value > 14:
        return {
            "success": False,
            "message": "Soil pH must be between 0 and 14."
        }

    # NPK values are collected but not scored yet.
    # Crop-specific nutrient reference ranges need validation.

    npk_provided = any(
        value is not None
        for value in [
            nitrogen_value,
            phosphorus_value,
            potassium_value
        ]
    )

    # ======================================
    # 4. CALCULATE HISTORICAL YIELD
    # ======================================

    recommendations = []

    for crop, crop_name in TARGET_CROPS.items():

        crop_data = filtered[
            filtered["Crop"] == crop
        ].copy()

        if crop_data.empty:
            continue

        total_area = crop_data["Area"].sum()
        total_production = crop_data["Production"].sum()

        if total_area <= 0:
            continue

        historical_yield = total_production / total_area

        if historical_yield < 0:
            continue

        valid_records = crop_data.dropna(
            subset=["Area", "Production"]
        )

        if valid_records.empty:
            continue

        profile = CROP_PROFILES[crop]

        # ==================================
        # 5. SUITABILITY SCREENING
        # ==================================


        # ==================================
        # 5. SUITABILITY SCORE BREAKDOWN
        # ==================================

        season_score = 0
        soil_score = 0
        water_score = 0
        ph_score = 0

        reasons = []
        warnings = []

        # Season: 30 points

        if season in profile["seasons"]:
            season_score = 30
            reasons.append("Suitable growing season")
        else:
            warnings.append("Season suitability not established")

        # Soil: 15 points

        if soil_type:
            if soil_type in profile["soils"]:
                soil_score = 15
                reasons.append("Soil type matches screening criteria")
            else:
                warnings.append("Soil type needs expert verification")
        else:
            warnings.append("Soil type not provided")

        # Water: 15 points

        if water:
            if water in profile["water"]:
                water_score = 15
                reasons.append("Water availability matches screening criteria")
            else:
                warnings.append("Water requirement needs verification")
        else:
            warnings.append("Water availability not provided")

        # pH: 20 points

        if ph_value is not None:
            if profile["ph_min"] <= ph_value <= profile["ph_max"]:
                ph_score = 20
                reasons.append("pH is within the screening range")
            else:
                warnings.append("pH is outside the screening range")
        else:
            warnings.append("Soil pH not provided")

        # NPK information status

        if npk_provided:
            reasons.append(
                "NPK values provided but not included in scoring"
            )
        else:
            warnings.append(
                "NPK values not provided; nutrient suitability not evaluated"
            )

        base_score = (
            season_score +
            soil_score +
            water_score +
            ph_score
        )

        recommendations.append({
            "crop": crop_name,
            "historical_name": crop,
            "yield": round(historical_yield, 2),
            "records": int(len(valid_records)),
            "years": int(valid_records["Crop_Year"].nunique()),

            "score": base_score,

            "score_breakdown": {
                "season": season_score,
                "soil": soil_score,
                "water": water_score,
                "ph": ph_score,
                "historical": 0
            },

            "reasons": reasons,
            "warnings": warnings
        })


    # ======================================
    # 6. HISTORICAL PERFORMANCE SCORE
    # ======================================

    yields = [
        item["yield"]
        for item in recommendations
    ]

    minimum_yield = min(yields)
    maximum_yield = max(yields)

    for item in recommendations:

        if maximum_yield > minimum_yield:

            yield_score = (
                (item["yield"] - minimum_yield) /
                (maximum_yield - minimum_yield)
            ) * 20

        else:
            yield_score = 10

        item["historical_score"] = round(
            yield_score, 1
        )

        item["score_breakdown"]["historical"] = round(
    yield_score, 1
)

        item["score"] = round(
            item["score"] + yield_score, 1
        )

    # ======================================
    # 7. RANK RECOMMENDATIONS
    # ======================================


    recommendations.sort(
        key=lambda item: (
            item["score"],
            item["yield"],
            item["records"]
        ),
        reverse=True
    )

    # Assign ranking and historical performance explanation

    for index, item in enumerate(recommendations, start=1):

        item["rank"] = index

        if item["historical_score"] >= 15:
            item["historical_explanation"] = (
                "Strong historical yield compared with other "
                "available crops in this district and season."
            )

        elif item["historical_score"] >= 8:
            item["historical_explanation"] = (
                "Moderate historical yield compared with other "
                "available crops in this district and season."
            )

        else:
            item["historical_explanation"] = (
                "Lower historical yield compared with other "
                "available crops in this district and season."
            )

    # Return only available top three crops

    top_recommendations = recommendations[:3]


    # ======================================
    # 8. RETURN RESULTS
    # ======================================

    return {
        "success": True,
        "state": state,
        "district": district,
        "season": season,
        "recommendations": top_recommendations,
        "npk_provided": npk_provided,
        "data_period": "1997–2010",
        "message": (
            "Scores combine historical yield and prototype "
            "season, soil, water and pH screening rules. "
            "NPK values are collected but are not included "
            "in scoring. These scores are not scientifically "
            "validated probabilities or guaranteed crop outcomes."
        )
    }
