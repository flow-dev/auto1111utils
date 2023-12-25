# MosaicImg

Python script (`MosaicImg.py`) for creating mosaic art from a target image using a collection of source images.

## Overview

This script creates mosaic art by arranging and adjusting a collection of source images to match a target image. The comparison is done based on color spaces (HSV, Hue, LAB) to find the best-matching pieces.

![mosic_generation_process_ref.gif](mosic_generation_process_ref.gif)

## Usage

1. Place the script (`MosaicImg.py`) in the directory containing source images.

2. Specify the source image directory (`input_dir`) and the target image path (`target_image_path`) in the script.

3. Optionally, adjust the comparison mode (`mode`), the number of divisions (`n_div`), and the piece scale (`piece_scale`).

4. Run the script:

   ```bash
   python MosaicImg.py
   ```

5. The script will create a mosaic art image and save it as `concat_img.jpg`.

## Dependencies

- NumPy
- OpenCV (cv2)
- Matplotlib
- tqdm
- Pillow (PIL)

## Configuration

Adjust the following parameters in the script according to your preferences:

- `input_dir`: Directory containing source images.
- `target_image_path`: Path to the target image.
- `mode`: Comparison mode (`HSV`, `Hue`, `LAB`, `LAB2`, `LAB4`).
- `n_div`: Number of divisions for creating the mosaic art.
- `piece_scale`: Scale factor for resizing the source images.

## Example

The following code creates mosaic art:

```python
python MosaicImg.py
```

## Output

The resulting mosaic art image will be saved as `concat_img.jpg` in the same directory.

## License

This project is licensed under the [MIT License](LICENSE.md). See the [LICENSE.md](LICENSE.md) file for details.
