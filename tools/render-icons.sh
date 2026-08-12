#!/bin/sh
# Regenerates the raster app icons from the SVG sources.
# Requires rsvg-convert (librsvg): brew install librsvg
set -e
cd "$(dirname "$0")/.."

rsvg-convert -w 512 -h 512 packages/app/public/icon.svg -o packages/app/public/icon-512.png
rsvg-convert -w 192 -h 192 packages/app/public/icon.svg -o packages/app/public/icon-192.png
rsvg-convert -w 1200 -h 630 packages/app/public/og-source.svg -o packages/app/public/og.png

# GitHub brand assets (uploaded by hand — GitHub has no API for avatars
# or repository social previews).
rsvg-convert -w 1024 -h 1024 docs/brand/avatar.svg -o docs/brand/avatar.png
rsvg-convert -w 1280 -h 640 docs/brand/social-preview.svg -o docs/brand/social-preview.png
echo "icons rendered"
