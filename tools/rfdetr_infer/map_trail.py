"""Write Leaflet map-trail HTML for GPS route + defect pins."""
from __future__ import annotations

import json
from pathlib import Path

# Hex colors aligned with class order in taxonomy
_CLASS_COLORS = {
    "alligator_crack": "#e74c3c",
    "drainage_issue": "#3498db",
    "longitudinal_crack": "#e67e22",
    "pothole": "#9b59b6",
    "ravelling": "#f1c40f",
    "edge_damage": "#1abc9c",
}


def write_map_trail(
    out_path: Path,
    *,
    route: list[dict],
    defects: list[dict],
    title: str = "RF-DETR near-field defect map",
) -> Path:
    """route: [{lat, lon, t, chainage_m?}]; defects: rows with lat/lon/class."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_gps = len(route) >= 2 or any(
        d.get("lat") is not None and d.get("lon") is not None for d in defects
    )

    center = [20.0, 77.0]
    zoom = 5
    if route:
        center = [route[len(route) // 2]["lat"], route[len(route) // 2]["lon"]]
        zoom = 16
    elif has_gps:
        for d in defects:
            if d.get("lat") is not None:
                center = [d["lat"], d["lon"]]
                zoom = 16
                break

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{ margin:0; height:100%; font-family: system-ui, sans-serif; }}
    #map {{ height: 100%; width: 100%; }}
    .banner {{
      position: absolute; z-index: 1000; top: 10px; left: 50px; right: 10px;
      background: rgba(255,255,255,0.92); padding: 8px 12px; border-radius: 6px;
      box-shadow: 0 1px 4px rgba(0,0,0,.2); font-size: 14px;
    }}
    .legend span {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }}
  </style>
</head>
<body>
  <div class="banner">
    <strong>{title}</strong>
    <div id="status"></div>
    <div class="legend" id="legend"></div>
  </div>
  <div id="map"></div>
  <script>
    const route = {json.dumps(route)};
    const defects = {json.dumps(defects)};
    const classColors = {json.dumps(_CLASS_COLORS)};
    const hasGps = {json.dumps(has_gps)};
    const map = L.map('map').setView({json.dumps(center)}, {zoom});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap'
    }}).addTo(map);

    const status = document.getElementById('status');
    if (!hasGps) {{
      status.textContent = 'No GPS / SRT — map has no route. See defects.csv for timestamps.';
    }} else {{
      status.textContent = route.length + ' GPS points, ' + defects.length + ' defects';
    }}

    if (route.length >= 2) {{
      const latlngs = route.map(p => [p.lat, p.lon]);
      const line = L.polyline(latlngs, {{ color: '#2c3e50', weight: 4, opacity: 0.75 }}).addTo(map);
      map.fitBounds(line.getBounds(), {{ padding: [30, 30] }});
      // fade trail: draw recent segment thicker
      const n = latlngs.length;
      if (n > 5) {{
        L.polyline(latlngs.slice(Math.floor(n * 0.7)), {{
          color: '#3498db', weight: 6, opacity: 0.9
        }}).addTo(map);
      }}
      L.circleMarker(latlngs[latlngs.length - 1], {{
        radius: 7, color: '#000', fillColor: '#fff', fillOpacity: 1
      }}).addTo(map).bindPopup('Current / end of track');
    }}

    const used = new Set();
    defects.forEach(d => {{
      if (d.lat == null || d.lon == null) return;
      const col = classColors[d.class] || '#e74c3c';
      used.add(d.class);
      L.circleMarker([d.lat, d.lon], {{
        radius: 8, color: '#222', weight: 1, fillColor: col, fillOpacity: 0.9
      }}).addTo(map).bindPopup(
        `<b>${{d.class}}</b><br/>id=${{d.defect_id}} conf=${{(d.conf_max || d.conf || 0).toFixed ? (d.conf_max || d.conf || 0).toFixed(2) : d.conf}}` +
        `<br/>t=${{d.t_start_s}}–${{d.t_end_s}}s` +
        (d.chainage_m != null ? `<br/>chainage=${{Number(d.chainage_m).toFixed(1)}} m` : '')
      );
    }});

    const legend = document.getElementById('legend');
    [...used].sort().forEach(name => {{
      const col = classColors[name] || '#999';
      const el = document.createElement('div');
      el.innerHTML = `<span style="background:${{col}}"></span>${{name}}`;
      legend.appendChild(el);
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
