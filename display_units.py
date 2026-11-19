"""Equivalent volume notation for presentation, never numeric conversion."""
import re


def display_volume_units(text):
    if not isinstance(text, str):
        return text
    replacements = {
        "мл": "см³", "мл/мин": "см³/мин", "мл/хв": "см³/хв",
        "мг/л": "мг/дм³", "л/г": "дм³/г", "л/мг": "дм³/мг",
        "л/мин": "дм³/мин", "л/хв": "дм³/хв", "л/ч": "дм³/ч", "л/год": "дм³/год",
        "mg/L": "mg/dm³", "L/g": "dm³/g", "L/mg": "dm³/mg",
        "ml/min": "cm³/min", "mL/min": "cm³/min", "L/h": "dm³/h", "L/min": "dm³/min",
        "mL": "cm³", "ml": "cm³", "л": "дм³",
    }
    pattern = r"(?<![\w])(" + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)) + r")(?![\w])"
    return re.sub(pattern, lambda match: replacements[match[0]], text)


def equivalent_ml_unit(unit):
    """Canonical aliases: equal-sized units carry no factor of 1000."""
    return {"cm3/min": "ml/min", "cm³/min": "ml/min", "см³/хв": "ml/min",
            "mg/dm3": "mg/L", "mg/dm³": "mg/L", "мг/дм³": "mg/L",
            "dm3/g": "L/g", "dm³/g": "L/g", "дм³/г": "L/g"}.get(unit, unit)
