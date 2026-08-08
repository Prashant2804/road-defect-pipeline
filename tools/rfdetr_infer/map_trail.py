"""Write synced video + map dashboard HTML (Google Maps or Leaflet)."""
from __future__ import annotations

import json
import os
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

_CLASS_LABELS = {
    "alligator_crack": "Alligator crack",
    "drainage_issue": "Drainage / water",
    "longitudinal_crack": "Long / transverse crack",
    "pothole": "Pothole",
    "ravelling": "Ravelling / rutting",
    "edge_damage": "Edge / shoulder",
}


def write_map_trail(
    out_path: Path,
    *,
    route: list[dict],
    defects: list[dict],
    title: str = "Road defect assessment",
    video_src: str = "annotated.mp4",
    z_far_m: float = 5.0,
    maps_api_key: str | None = None,
) -> Path:
    """Write a 3-panel dashboard: stats | annotated video | map (synced).

    ``maps_api_key`` — Google Maps JavaScript API key. If empty, falls back to
    Leaflet/OSM. Prefer env ``GOOGLE_MAPS_API_KEY`` at call sites.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    key = (maps_api_key or os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
    use_google = bool(key)

    has_gps = len(route) >= 2 or any(
        d.get("lat") is not None and d.get("lon") is not None for d in defects
    )

    center = [20.0, 77.0]
    if route:
        mid = route[len(route) // 2]
        center = [mid["lat"], mid["lon"]]
    else:
        for d in defects:
            if d.get("lat") is not None and d.get("lon") is not None:
                center = [d["lat"], d["lon"]]
                break

    # Slim defect payload for the browser
    slim_defects = []
    for d in defects:
        slim_defects.append(
            {
                "id": d.get("defect_id"),
                "class": d.get("class"),
                "conf": d.get("conf_max", d.get("conf")),
                "t0": d.get("t_start_s"),
                "t1": d.get("t_end_s"),
                "lat": d.get("lat"),
                "lon": d.get("lon"),
                "chainage_m": d.get("chainage_m"),
            }
        )

    payload = {
        "title": title,
        "videoSrc": video_src,
        "zFarM": z_far_m,
        "route": route,
        "defects": slim_defects,
        "classColors": _CLASS_COLORS,
        "classLabels": _CLASS_LABELS,
        "hasGps": has_gps,
        "center": center,
        "useGoogle": use_google,
    }

    maps_script = ""
    if use_google:
        maps_script += (
            f'<script src="https://maps.googleapis.com/maps/api/js?key={key}"></script>\n'
        )
    # Always ship Leaflet as offline fallback if Google fails / no key
    maps_script += """
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  {maps_script}
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1a1d21;
      --muted: #5c6570;
      --line: #e2e6ea;
      --accent: #0d9488;
      --font: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
      --mono: "IBM Plex Mono", ui-monospace, monospace;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; height: 100%;
      background: var(--bg); color: var(--ink);
      font-family: var(--font);
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(220px, 280px) minmax(0, 1.4fr) minmax(280px, 1fr);
      grid-template-rows: auto 1fr;
      height: 100%;
      gap: 0;
    }}
    header.top {{
      grid-column: 1 / -1;
      display: flex; align-items: baseline; justify-content: space-between;
      padding: 12px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    header.top h1 {{
      margin: 0; font-size: 16px; font-weight: 650; letter-spacing: -0.01em;
    }}
    header.top .sub {{ color: var(--muted); font-size: 12px; }}
    aside.stats, section.video-pane, section.map-pane {{
      background: var(--panel);
      border-right: 1px solid var(--line);
      display: flex; flex-direction: column;
      min-height: 0;
    }}
    section.map-pane {{ border-right: none; }}
    .pane-head {{
      padding: 12px 16px 8px;
      border-bottom: 1px solid var(--line);
    }}
    .pane-head h2 {{
      margin: 0; font-size: 13px; font-weight: 650; text-transform: none;
    }}
    .pane-head p {{
      margin: 4px 0 0; font-size: 11px; color: var(--muted); line-height: 1.35;
    }}
    .stats-body {{
      padding: 16px; overflow: auto; flex: 1;
    }}
    .metric {{
      margin-bottom: 18px;
    }}
    .metric .label {{
      font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase;
      color: var(--muted); font-weight: 600;
    }}
    .metric .value {{
      font-family: var(--mono); font-size: 28px; font-weight: 600;
      letter-spacing: -0.03em; line-height: 1.1; margin-top: 2px;
    }}
    .metric .hint {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
    .legend {{ margin-top: 8px; }}
    .legend-row {{
      display: flex; align-items: center; gap: 10px;
      padding: 8px 0; border-top: 1px solid var(--line);
      font-size: 12px;
    }}
    .legend-row .swatch {{
      width: 4px; height: 28px; border-radius: 2px; flex-shrink: 0;
    }}
    .legend-row .name {{ flex: 1; color: var(--ink); }}
    .legend-row .amt {{
      font-family: var(--mono); font-size: 12px; color: var(--muted);
    }}
    .progress {{
      margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--line);
    }}
    .progress .bar {{
      height: 6px; background: #e8ecef; border-radius: 99px; overflow: hidden;
    }}
    .progress .fill {{
      height: 100%; width: 0%; background: var(--accent); border-radius: 99px;
    }}
    .progress .meta {{
      display: flex; justify-content: space-between;
      font-size: 11px; color: var(--muted); margin-top: 6px;
      font-family: var(--mono);
    }}
    .warn {{
      margin-top: 12px; padding: 10px; background: #fff7ed;
      border: 1px solid #fed7aa; border-radius: 8px;
      font-size: 12px; color: #9a3412; line-height: 1.4;
    }}
    .video-wrap {{
      flex: 1; min-height: 0; display: flex; align-items: center;
      justify-content: center; background: #0b0d10; padding: 8px;
    }}
    video {{
      max-width: 100%; max-height: 100%; width: 100%;
      background: #000; border-radius: 4px;
    }}
    #map {{
      flex: 1; min-height: 0; width: 100%;
      background: #dbe4ea;
    }}
    @media (max-width: 980px) {{
      .app {{
        grid-template-columns: 1fr;
        grid-template-rows: auto auto 42vh 42vh;
        height: auto; min-height: 100%;
      }}
      aside.stats, section.video-pane, section.map-pane {{
        border-right: none; border-bottom: 1px solid var(--line);
      }}
      .video-wrap {{ min-height: 40vh; }}
      #map {{ min-height: 40vh; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header class="top">
      <h1 id="title">{title}</h1>
      <div class="sub" id="backend-label"></div>
    </header>

    <aside class="stats">
      <div class="pane-head">
        <h2>Assessment</h2>
        <p>Synced to video time · SRT GPS when available</p>
      </div>
      <div class="stats-body">
        <div class="metric">
          <div class="label">Position</div>
          <div class="value" id="m-pos">—</div>
          <div class="hint">m along route</div>
        </div>
        <div class="metric">
          <div class="label">In view</div>
          <div class="value" id="m-view">—</div>
          <div class="hint" id="m-view-hint">near-field detections</div>
        </div>
        <div class="metric">
          <div class="label">Cumulative</div>
          <div class="value" id="m-cum">—</div>
          <div class="hint">m total length</div>
        </div>
        <div class="legend" id="legend"></div>
        <div class="progress">
          <div class="bar"><div class="fill" id="prog-fill"></div></div>
          <div class="meta">
            <span id="prog-t">0 / 0 s</span>
            <span id="prog-m">0 / 0 m</span>
          </div>
        </div>
        <div class="warn" id="gps-warn" hidden></div>
      </div>
    </aside>

    <section class="video-pane">
      <div class="pane-head">
        <h2>Detection overlay · <span id="z-far-label">5</span> m ahead</h2>
        <p>Green wash = assess corridor · boxes = defects in near field</p>
      </div>
      <div class="video-wrap">
        <video id="vid" controls playsinline preload="metadata">
          <source src="{video_src}" type="video/mp4"/>
          Annotated video not found next to this HTML.
        </video>
      </div>
    </section>

    <section class="map-pane">
      <div class="pane-head">
        <h2>Map · north-up · live trail</h2>
        <p>Past detections solid · upcoming faded · vehicle follows GPS</p>
      </div>
      <div id="map"></div>
    </section>
  </div>

  <script>
    const DATA = {json.dumps(payload)};
    document.getElementById('title').textContent = DATA.title;
    document.getElementById('z-far-label').textContent = String(DATA.zFarM);
    document.getElementById('backend-label').textContent =
      DATA.useGoogle ? 'Google Maps' : 'Leaflet / OpenStreetMap';

    const vid = document.getElementById('vid');
    const legendEl = document.getElementById('legend');
    const warnEl = document.getElementById('gps-warn');

    if (!DATA.hasGps) {{
      warnEl.hidden = false;
      warnEl.textContent =
        'No GPS route in this run (SRT missing or unparsed). ' +
        'Rebuild with a GoPro .SRT: python -m tools.rfdetr_infer.rebuild_dashboard --run-dir <dir> --srt path.SRT';
    }}

    // Class totals (counts; chainage span when available)
    const classOrder = Object.keys(DATA.classColors);
    const totals = {{}};
    classOrder.forEach(c => totals[c] = 0);
    DATA.defects.forEach(d => {{
      if (d.class in totals) totals[d.class] += 1;
    }});
    classOrder.forEach(c => {{
      if (!totals[c]) return;
      const row = document.createElement('div');
      row.className = 'legend-row';
      row.innerHTML =
        `<div class="swatch" style="background:${{DATA.classColors[c]}}"></div>` +
        `<div class="name">${{DATA.classLabels[c] || c}}</div>` +
        `<div class="amt">${{totals[c]}}</div>`;
      legendEl.appendChild(row);
    }});

    const maxChain = (() => {{
      let m = 0;
      DATA.route.forEach(p => {{ if (p.chainage_m != null) m = Math.max(m, p.chainage_m); }});
      DATA.defects.forEach(d => {{ if (d.chainage_m != null) m = Math.max(m, d.chainage_m); }});
      return m;
    }})();

    function lerpRoute(t) {{
      const r = DATA.route;
      if (!r.length) return null;
      if (t <= r[0].t) return r[0];
      if (t >= r[r.length - 1].t) return r[r.length - 1];
      for (let i = 0; i < r.length - 1; i++) {{
        const a = r[i], b = r[i + 1];
        if (t >= a.t && t <= b.t) {{
          const u = (t - a.t) / Math.max(b.t - a.t, 1e-6);
          return {{
            lat: a.lat + (b.lat - a.lat) * u,
            lon: a.lon + (b.lon - a.lon) * u,
            t,
            chainage_m: (a.chainage_m != null && b.chainage_m != null)
              ? a.chainage_m + (b.chainage_m - a.chainage_m) * u
              : (a.chainage_m ?? b.chainage_m ?? null),
          }};
        }}
      }}
      return r[r.length - 1];
    }}

    // ---- Map backends ----
    let mapApi = null;

    function initGoogle() {{
      const map = new google.maps.Map(document.getElementById('map'), {{
        center: {{ lat: DATA.center[0], lng: DATA.center[1] }},
        zoom: DATA.hasGps ? 17 : 5,
        mapTypeId: 'roadmap',
        disableDefaultUI: false,
        zoomControl: true,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: true,
      }});
      const path = DATA.route.map(p => ({{ lat: p.lat, lng: p.lon }}));
      let routeLine = null;
      if (path.length >= 2) {{
        routeLine = new google.maps.Polyline({{
          path, strokeColor: '#334155', strokeOpacity: 0.55, strokeWeight: 4, map
        }});
        const b = new google.maps.LatLngBounds();
        path.forEach(p => b.extend(p));
        map.fitBounds(b, 40);
      }}
      const vehicle = new google.maps.Marker({{
        map,
        position: path[0] || {{ lat: DATA.center[0], lng: DATA.center[1] }},
        icon: {{
          path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
          scale: 5, fillColor: '#0f172a', fillOpacity: 1,
          strokeColor: '#fff', strokeWeight: 1, rotation: 0,
        }},
      }});
      const markers = DATA.defects.map(d => {{
        if (d.lat == null || d.lon == null) return null;
        const m = new google.maps.Marker({{
          map,
          position: {{ lat: d.lat, lng: d.lon }},
          opacity: 0.25,
          icon: {{
            path: google.maps.SymbolPath.CIRCLE,
            scale: 7,
            fillColor: DATA.classColors[d.class] || '#e74c3c',
            fillOpacity: 0.95,
            strokeColor: '#111', strokeWeight: 1,
          }},
          title: `${{d.class}} @ ${{d.t0}}s`,
        }});
        return {{ d, m }};
      }}).filter(Boolean);

      return {{
        setVehicle(lat, lon, headingDeg) {{
          vehicle.setPosition({{ lat, lng: lon }});
          if (headingDeg != null) {{
            const icon = vehicle.getIcon();
            icon.rotation = headingDeg;
            vehicle.setIcon(icon);
          }}
          // Keep vehicle roughly in view without fighting user pan too hard
          const c = map.getCenter();
          if (!c) map.panTo({{ lat, lng: lon }});
        }},
        setDefectVisibility(pred) {{
          markers.forEach(({{ d, m }}) => {{
            const state = pred(d); // 'past' | 'now' | 'future'
            m.setOpacity(state === 'future' ? 0.2 : state === 'now' ? 1 : 0.85);
            m.setZIndex(state === 'now' ? 999 : 1);
          }});
        }},
      }};
    }}

    function initLeaflet() {{
      const map = L.map('map').setView(DATA.center, DATA.hasGps ? 17 : 5);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 20, attribution: '&copy; OpenStreetMap'
      }}).addTo(map);
      if (DATA.route.length >= 2) {{
        const latlngs = DATA.route.map(p => [p.lat, p.lon]);
        const line = L.polyline(latlngs, {{ color: '#334155', weight: 4, opacity: 0.55 }}).addTo(map);
        map.fitBounds(line.getBounds(), {{ padding: [30, 30] }});
      }}
      const vehicle = L.circleMarker(DATA.center, {{
        radius: 8, color: '#fff', weight: 2, fillColor: '#0f172a', fillOpacity: 1
      }}).addTo(map);
      const markers = DATA.defects.map(d => {{
        if (d.lat == null || d.lon == null) return null;
        const m = L.circleMarker([d.lat, d.lon], {{
          radius: 7, color: '#111', weight: 1,
          fillColor: DATA.classColors[d.class] || '#e74c3c',
          fillOpacity: 0.9, opacity: 0.25
        }}).addTo(map);
        m.bindPopup(`<b>${{d.class}}</b><br/>t=${{d.t0}}–${{d.t1}}s`);
        return {{ d, m }};
      }}).filter(Boolean);
      return {{
        setVehicle(lat, lon) {{
          vehicle.setLatLng([lat, lon]);
        }},
        setDefectVisibility(pred) {{
          markers.forEach(({{ d, m }}) => {{
            const state = pred(d);
            m.setStyle({{ opacity: state === 'future' ? 0.2 : 1 }});
          }});
        }},
      }};
    }}

    if (DATA.useGoogle && window.google && google.maps) {{
      mapApi = initGoogle();
    }} else {{
      mapApi = initLeaflet();
      if (DATA.useGoogle) {{
        warnEl.hidden = false;
        warnEl.textContent += ' (Google Maps script failed — using Leaflet.)';
      }}
    }}

    function headingFromRoute(t) {{
      const r = DATA.route;
      if (r.length < 2) return null;
      let i = 0;
      while (i < r.length - 2 && r[i + 1].t < t) i++;
      const a = r[i], b = r[Math.min(i + 1, r.length - 1)];
      const dLon = (b.lon - a.lon) * Math.cos((a.lat + b.lat) * Math.PI / 360);
      const dLat = b.lat - a.lat;
      return Math.atan2(dLon, dLat) * 180 / Math.PI;
    }}

    function tick() {{
      const t = vid.currentTime || 0;
      const dur = vid.duration || Math.max(t, 1);
      const pos = lerpRoute(t);
      const chain = pos && pos.chainage_m != null ? pos.chainage_m : null;

      document.getElementById('m-pos').textContent =
        chain != null ? chain.toFixed(0) : (t.toFixed(1) + 's');
      document.getElementById('m-cum').textContent =
        maxChain > 0 ? maxChain.toFixed(1) : '—';

      // In-view: defects whose time window overlaps [t-1, t+1]
      let inView = 0;
      DATA.defects.forEach(d => {{
        const t0 = d.t0 ?? 0, t1 = d.t1 ?? t0;
        if (t1 >= t - 0.75 && t0 <= t + 0.75) inView += 1;
      }});
      document.getElementById('m-view').textContent = String(inView);
      document.getElementById('m-view-hint').textContent =
        `detections near t=${{t.toFixed(1)}}s · assess ≤ ${{DATA.zFarM}} m`;

      const pct = Math.min(100, (t / dur) * 100);
      document.getElementById('prog-fill').style.width = pct + '%';
      document.getElementById('prog-t').textContent =
        `${{t.toFixed(0)}} / ${{isFinite(dur) ? dur.toFixed(0) : '—'}} s`;
      document.getElementById('prog-m').textContent =
        `${{chain != null ? chain.toFixed(0) : 0}} / ${{maxChain > 0 ? maxChain.toFixed(0) : '—'}} m`;

      if (mapApi && pos) {{
        mapApi.setVehicle(pos.lat, pos.lon, headingFromRoute(t));
      }}
      if (mapApi) {{
        mapApi.setDefectVisibility(d => {{
          const t0 = d.t0 ?? 0;
          if (t0 > t + 0.3) return 'future';
          if (Math.abs((d.t1 ?? t0) - t) < 1.0 || (t0 <= t && (d.t1 ?? t0) >= t - 0.5))
            return 'now';
          return 'past';
        }});
      }}
    }}

    vid.addEventListener('timeupdate', tick);
    vid.addEventListener('loadedmetadata', tick);
    vid.addEventListener('seeked', tick);
    setInterval(tick, 250);
    tick();
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
