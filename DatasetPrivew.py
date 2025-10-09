import os
from PIL import Image

# CONFIGURATION
ROOT_DIR = "Data/Bulgarian"  # Parent dir with subfolders
OUTPUT_NAME = "wide_collage.jpg"
THUMBNAIL_SIZE = (64, 64)  # Smaller thumbnails for more columns
PADDING = 2                # Minimal spacing (set to 0 for grid-like)
GRID_SIZE = (5, 15)        # 5 folders (rows) x 15 images (columns)
BACKGROUND_COLOR = (255, 255, 255)  # White

def create_wide_collage():
    # Calculate total collage size (account for padding)
    cell_width = THUMBNAIL_SIZE[0] + PADDING
    cell_height = THUMBNAIL_SIZE[1] + PADDING
    collage_width = cell_width * GRID_SIZE[1] + PADDING
    collage_height = cell_height * GRID_SIZE[0] + PADDING
    collage = Image.new('RGB', (collage_width, collage_height), BACKGROUND_COLOR)

    # Get first N folders (alphabetically)
    folders = sorted([f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))])[:GRID_SIZE[0]]

    for row, folder in enumerate(folders):
        folder_path = os.path.join(ROOT_DIR, folder)
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])[:GRID_SIZE[1]]

        for col, img_name in enumerate(images):
            try:
                img = Image.open(os.path.join(folder_path, img_name))
                img.thumbnail(THUMBNAIL_SIZE)  # Keeps aspect ratio
                x = col * cell_width + PADDING
                y = row * cell_height + PADDING
                collage.paste(img, (x, y))
            except Exception as e:
                print(f"Skipping {img_name} (error: {e})")

    collage.save(OUTPUT_NAME)
    print(f"Wide collage saved as {OUTPUT_NAME} (Size: {collage.size[0]}x{collage.size[1]}px)")

if __name__ == "__main__":
    create_wide_collage()