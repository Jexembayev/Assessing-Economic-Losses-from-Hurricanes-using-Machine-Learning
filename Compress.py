import os
import argparse

def compress_tif(original_tif, compression_method="JPEG", predictors=2, new_directory="compressed/"):
    """
    This function takes an uncompressed GeoTIFF and compresses it with one of four compression methods:
        - Packbits
        - JPEG
        - Deflate
        - LZW
    For LZW and Deflate, you can choose the number of predictors.

    :param original_tif: The uncompressed GeoTIFF to be compressed
    :param compression_method: Packbits, JPEG, Deflate, or LZW
    :param predictors: Default is 2
    :param new_directory: Directory to save the compressed TIF
    :return: Creates a new compressed TIF in the specified directory
    """

    new_tif_base = os.path.splitext(original_tif)[0]
    output_files = {
        "Packbits": f"{new_tif_base}_packbit_compressed.tif",
        "JPEG": f"{new_tif_base}_jpeg_compressed.tif",
        "Deflate": f"{new_tif_base}_deflate_compressed.tif",
        "LZW": f"{new_tif_base}_lzw_compressed.tif"
    }

    commands = {
        "Packbits": f"gdal_translate -of GTiff -co COMPRESS=PACKBITS -co TILED=YES {original_tif} {output_files['Packbits']}",
        "JPEG": f"gdal_translate -co COMPRESS=JPEG -co TILED=YES {original_tif} {output_files['JPEG']}",
        "Deflate": f"gdal_translate -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR={predictors} -co TILED=YES {original_tif} {output_files['Deflate']}",
        "LZW": f"gdal_translate -of GTiff -co COMPRESS=LZW -co PREDICTOR={predictors} -co TILED=YES {original_tif} {output_files['LZW']}"
    }

    if compression_method in commands:
        os.system(commands[compression_method])
        os.system(f"mv {output_files[compression_method]} {new_directory}")
    else:
        raise ValueError("Invalid compression method. Choose from: Packbits, JPEG, Deflate, LZW.")

def compress_directory(directory_name, new_directory, compression_method="JPEG", predictors=2):
    """
    Compresses a directory of TIFFs.

    :param directory_name: Directory containing uncompressed TIFFs
    :param new_directory: Directory to save compressed TIFs
    :param compression_method: JPEG, Packbits, Deflate, or LZW
    :param predictors: Default is 2
    :return: A new directory populated with compressed TIFFs
    """

    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    for filename in os.listdir(directory_name):
        file_path = os.path.join(directory_name, filename)
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            compress_tif(file_path, compression_method, predictors, new_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress GeoTIFF files in a directory.",
                                     epilog='Run this script from the parent directory of the TIFs.')
    parser.add_argument('directory', help='The directory containing the TIF files')
    parser.add_argument('new_directory', help='The directory to save the compressed TIF files')
    parser.add_argument('method', choices=["JPEG", "Packbits", "Deflate", "LZW"], help='Compression method')
    parser.add_argument('--predictors', type=int, default=2, help='Predictors for Deflate and LZW (default: 2)')
    args = parser.parse_args()

    compress_directory(args.directory, args.new_directory, args.method, args.predictors)
