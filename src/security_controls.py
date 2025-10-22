def suggest_controls(level):
    """Return mitigation strategies per risk level"""
    controls = {
        "Low": ["Use antivirus", "Apply OS patches regularly"],
        "Medium": ["Enable IDS/IPS", "Enforce role-based access"],
        "High": ["Network segmentation", "Zero Trust framework", "Incident Response Plan"]
    }
    return controls.get(level, ["No recommendation available"])
