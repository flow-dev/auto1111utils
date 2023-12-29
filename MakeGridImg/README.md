# MakeGridImg

Python script (`MakeGridImg.py`) for creating a grid image from a collection of PNG images.

## Overview

This script reads all PNG files in the current directory, resizes them to a fixed size, and arranges them in a grid pattern. The resulting grid image is saved as a JPEG file.

![output_grid_20231229135804_9_images.jpg](output_grid_20231229135804_9_images.jpg)

## Usage

1. Place the script (`MakeGridImg.py`) in the directory containing PNG images.

2. Run the script:

   ```bash
   python MakeGridImg.py
   ```

3. The script will create a grid image (`output_grid_timestamp_num_images.jpg`) in the same directory.

## Dependencies

- NumPy
- Pillow (PIL)

## File Structure

```
/Your_Project_Directory
    ├── MakeGridImg.py
    └── image1.png
    └── image2.png
    └── ...
```

## Result

The script will create a grid image with the resized PNG files arranged in a grid pattern.

## Example

The following code creates a grid image from PNG files:

```python
python MakeGridImg.py
```

## Output

The resulting grid image will be saved in the same directory with a filename like `output_grid_timestamp_num_images.jpg`.

## License

This project is licensed under the [MIT License](LICENSE.md). See the [LICENSE.md](LICENSE.md) file for details.
