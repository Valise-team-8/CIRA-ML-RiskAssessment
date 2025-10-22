import pandas as pd

STRIDE_MAP = {
    "S": "Spoofing",
    "T": "Tampering",
    "R": "Repudiation",
    "I": "Information Disclosure",
    "D": "Denial of Service",
    "E": "Elevation of Privilege"
}

def generate_threat_profile(df):
    """Adds synthetic threat types for ML correlation"""
    stride_keys = list(STRIDE_MAP.keys())
    df["Vulnerability_Type"] = [stride_keys[i % len(stride_keys)] for i in range(len(df))]
    df["Threat_Category"] = df["Vulnerability_Type"].map(STRIDE_MAP)
    summary = df["Threat_Category"].value_counts().to_dict()
    print("[INFO] STRIDE threat modeling added.")
    return df, summary
