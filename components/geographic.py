import folium
import streamlit.components.v1 as components
import branca.colormap as cm

import json
import pandas as pd

def render_folium_html(m: folium.Map, height: int = 550):
    html = m.get_root().render()
    components.html(html, height=height, scrolling=True)
    
def make_choropleth_map(stats: pd.DataFrame, geojson: dict, selected_metric: str, measure="Mean"):
    
    m = folium.Map(
        location=[35.24, -80.84], 
        zoom_start=9, 
        tiles="CartoDB positron",
        zoom_control=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        touchZoom=False,
        boxZoom=False,
        dragging=False
    )
    
    vmin = stats[measure].min()
    vmax = stats[measure].max()

    colormap = cm.LinearColormap(
        colors=cm.linear.RdYlGn_11.colors,
        vmin=vmin,
        vmax=vmax,
        caption=selected_metric,
    )
    
    stats[measure] = round(stats[measure],2)
    
    measure_by_geoid = dict(zip(stats["geoid"], stats[measure]))
    
    for feature in geojson["features"]:
        props = feature["properties"]
        geoid = props.get("geoid")
        value = measure_by_geoid.get(geoid)
        feature["properties"][measure] = value
        
        name = props.get("name")
        
        lat = float(props.get("intptlat"))
        lon = float(props.get("intptlon"))
        
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 12px;
                    font-weight: 600;
                    color: black;
                    text-shadow: 1px 1px 2px white;
                ">
                    {name}
                </div>
                """
            )
        ).add_to(m)
    
    
    
    def county_outline_style(feature):
        value = feature["properties"].get(measure)
        return {
            "fillColor": colormap(value) if value is not None else "#cccccc",
            "fillOpacity": 0.0,
            "weight": 2,
            "fillOpacity": 0.8,
        }

    def county_highlight_style(feature):
        return {
            "weight": 3,
            "color": "black",
            "fillOpacity": 0.05,
        }
    
    # Add a crisp outline + hover tooltip
    folium.GeoJson(
        geojson,
        style_function=county_outline_style,
        highlight_function=county_highlight_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", measure],
            aliases=["Name",selected_metric]
        ),
    ).add_to(m)
    
    
    # colormap.add_to(m)
    
    return m

def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def build_geo_name_to_geoid(name_field, geoid_field, geojson_data: dict) -> dict:
    name_to_geoid = {}
    for f in geojson_data.get("features", []):
        props = f.get("properties", {})
        # 'name' = "Chester", 'geoid' = "45023"
        nm = str(props.get(name_field, "")).strip().upper()
        geoid = str(props.get(geoid_field, "")).strip()
        if nm and geoid:
            name_to_geoid[nm] = geoid
    return name_to_geoid
