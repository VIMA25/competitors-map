#!/usr/bin/env bash
set -e

echo "🏥 Setting up South Florida Competition Analysis"

# 0. cd to the folder that contains this script
cd "$(dirname "$0")"

# 1. check python3
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 not found"; exit 1; }

# 2. make & activate venv if not active
if [ -z "$VIRTUAL_ENV" ]; then
    [ -d venv ] || python3 -m venv venv
    source venv/bin/activate
fi

# 3. upgrade pip & install core dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. install Excel engine libraries
pip install xlrd openpyxl

# 5. final hint about Mapbox token
if [ -z "$MAPBOX_TOKEN" ] && ! grep -q MAPBOX_TOKEN .env 2>/dev/null; then
  echo "⚠️  No MAPBOX_TOKEN found – the app will fall back to free tiles."
  echo "   Add a .env file or export MAPBOX_TOKEN for Mapbox styling."
fi

echo "🚀 Launching Streamlit …  (Ctrl+C to stop)"
exec streamlit run Competitors_Map.py
