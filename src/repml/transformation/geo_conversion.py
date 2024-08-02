import pyproj


def convert_etrs89_to_latlon(easting, northing, zone_number, zone_letter):
    """Konvertiert ETRS89 Koordinaten in Längen- und Breitengrade.

    Args:
        easting (pd.Series): Rechtswert (x-Koordinate)
        northing (pd.Series): Hochwert (y-Koordinate)
        zone_number (int): UTM Zone
        zone_letter (str): UTM Hemisphäre

    Returns:
        tuple: Längen- und Breitengrade
    """
    # Definition der Projektionen
    utm_proj = pyproj.CRS.from_dict(
        {"proj": "utm", "zone": zone_number, "ellps": "WGS84", "hemisphere": zone_letter}
    )
    latlon_proj = pyproj.CRS.from_dict({"proj": "latlong", "ellps": "WGS84"})

    # Definition des Transformers + Konversion
    transformer = pyproj.Transformer.from_crs(utm_proj, latlon_proj)
    lon, lat = transformer.transform(easting, northing)
    return lon, lat
