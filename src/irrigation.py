"""Simple daily irrigation advisory in millimetres."""

from datetime import date


# Illustrative crop coefficients; calibrate for local conditions.
DEFAULT_KC = {
    "wheat": 1.0,
    "rice": 1.1,
    "maize": 1.0,
    "cotton": 1.0,
    "sugarcane": 1.1,
    "groundnut": 0.9,
    "soybean": 1.0,
    "potato": 1.0,
}

STAGE_MULTIPLIER = {
    "initial": 0.7,
    "development": 0.9,
    "mid": 1.0,
    "late": 0.8,
}


def irrigation_advisory(
    crop,
    et0_mm,
    forecast_rain_mm=0.0,
    effective_rain_fraction=0.8,
    irrigation_efficiency=0.85,
    kc=None,
    growth_stage="mid",
    soil_moisture_deficit_mm=None,
    forecast_temperature_c=None,
    forecast_humidity_pct=None,
    advisory_date=None,
):
    """
    Estimate irrigation depth.

    et0_mm: reference evapotranspiration in mm/day, preferably
            from a weather service or a validated ET0 calculation.
    forecast_rain_mm: expected rainfall in mm for the same period.
    soil_moisture_deficit_mm: optional measured/modelled soil water
            deficit. When supplied, it limits the net refill depth.
    """
    values = {
        "et0_mm": et0_mm,
        "forecast_rain_mm": forecast_rain_mm,
        "effective_rain_fraction": effective_rain_fraction,
        "irrigation_efficiency": irrigation_efficiency,
    }

    for name, value in values.items():
        if value is None or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a numeric value.")
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")

    if not 0 <= effective_rain_fraction <= 1:
        raise ValueError("effective_rain_fraction must be between 0 and 1.")

    if not 0 < irrigation_efficiency <= 1:
        raise ValueError("irrigation_efficiency must be in (0, 1].")

    crop_key = str(crop).strip().casefold()
    crop_kc = kc if kc is not None else DEFAULT_KC.get(crop_key)

    if crop_kc is None:
        raise ValueError(
            f"No default Kc for '{crop}'. Pass a locally validated kc."
        )
    if crop_kc <= 0:
        raise ValueError("kc must be greater than zero.")

    stage = str(growth_stage).strip().casefold()
    if stage not in STAGE_MULTIPLIER:
        raise ValueError(
            f"growth_stage must be one of {list(STAGE_MULTIPLIER)}."
        )

    kc_adjusted = crop_kc * STAGE_MULTIPLIER[stage]
    crop_water_use = kc_adjusted * et0_mm

    effective_rain = forecast_rain_mm * effective_rain_fraction
    net_need = max(0.0, crop_water_use - effective_rain)

    if soil_moisture_deficit_mm is not None:
        if soil_moisture_deficit_mm < 0:
            raise ValueError("Soil moisture deficit cannot be negative.")
        net_need = min(net_need, soil_moisture_deficit_mm)

    gross_need = net_need / irrigation_efficiency

    if net_need < 0.5:
        recommendation = (
            "No irrigation indicated by this estimate; check field soil moisture."
        )
    elif net_need < 5:
        recommendation = (
            "Small irrigation may be sufficient; verify soil moisture first."
        )
    else:
        recommendation = (
            "Irrigation may be needed. Confirm soil moisture and local "
            "crop-water guidance before applying water."
        )

    return {
        "date": str(advisory_date or date.today()),
        "crop": crop,
        "growth_stage": stage,
        "et0_mm_day": round(float(et0_mm), 2),
        "crop_water_use_mm_day": round(crop_water_use, 2),
        "forecast_rain_mm": round(float(forecast_rain_mm), 2),
        "effective_rain_mm": round(effective_rain, 2),
        "net_irrigation_mm": round(net_need, 2),
        "gross_irrigation_mm": round(gross_need, 2),
        "recommendation": recommendation,
        "status": (
            "Check soil moisture"
            if net_need < 0.5
            else "Potential irrigation need"
        ),
        "inputs_used": {
            "temperature_c": forecast_temperature_c,
            "humidity_pct": forecast_humidity_pct,
        },
        "warning": (
            "Estimate only. Temperature and humidity are recorded for "
            "context; ET0 must come from a valid calculation or data source. "
            "Rainfall, soil moisture, growth stage and local conditions "
            "affect the actual irrigation requirement."
        ),
    }


if __name__ == "__main__":
    result = irrigation_advisory(
        crop="wheat",
        et0_mm=4.5,
        forecast_rain_mm=0,
        growth_stage="mid",
        forecast_temperature_c=35,
        forecast_humidity_pct=40,
    )

    for key, value in result.items():
        print(f"{key}: {value}")