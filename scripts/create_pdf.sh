#!/usr/bin/env bash

# A script to create a PDF from a directory of images, with a generated title page.
# Usage: ./create_pdf.sh <output_pdf> <title> <image_directory> [resize_percentage]

set -e

# --- Arguments ---
OUTPUT_PDF="$1"
TITLE="$2"
IMAGE_DIR="$3"
RESIZE_PERCENTAGE="$4"

# --- Validation ---
if [ -z "$OUTPUT_PDF" ] || [ -z "$TITLE" ] || [ -z "$IMAGE_DIR" ]; then
  echo "Usage: $0 <output_pdf> <title> <image_directory> [resize_percentage]"
  exit 1
fi

if ! ls "$IMAGE_DIR"/*.png 1>/dev/null 2>&1; then
  echo "Notice: No images found in '$IMAGE_DIR', skipping PDF creation."
  exit 0
fi

echo "Creating PDF: $OUTPUT_PDF"

# --- Dynamic Page Setup ---
FIRST_PAGE=$(ls -A "$IMAGE_DIR"/*.png | head -n 1)
PAGE_SIZE=$(identify -format "%wx%h" "$FIRST_PAGE")
DENSITY=$(identify -format "%xx%y" "$FIRST_PAGE")
UNITS=$(identify -format "%u" "$FIRST_PAGE")

if [ -z "$UNITS" ] || [ "$UNITS" == "Undefined" ]; then
    UNITS="PixelsPerInch"
fi

# --- Resize Option ---
RESIZE_OPTS=""
if [ -n "$RESIZE_PERCENTAGE" ]; then
  echo "  > Resizing to $RESIZE_PERCENTAGE%"
  RESIZE_OPTS="-resize $RESIZE_PERCENTAGE%"
fi

echo "  > Detected page size: $PAGE_SIZE"
echo "  > Detected density: $DENSITY"
echo "  > Detected units: $UNITS"
echo "  > Title: $TITLE"

# --- PDF Creation ---
convert \
  \( -size "$PAGE_SIZE" -background white -fill black -gravity center \
     -units "$UNITS" -density "$DENSITY" -pointsize 48 label:"$TITLE" \) \
  -units "$UNITS" -density "$DENSITY" \
  "$IMAGE_DIR"/*.png \
  $RESIZE_OPTS \
  "$OUTPUT_PDF"

echo "PDF created successfully: $OUTPUT_PDF"
ls -lh "$OUTPUT_PDF"
