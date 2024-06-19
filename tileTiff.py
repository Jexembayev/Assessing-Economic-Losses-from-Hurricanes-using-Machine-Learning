import argparse
import os


def tile_tiff(tiff_file):

    command0 = "mkdir image_tiles"
    command1 = "gdal_retile.py -v -r bilinear -levels 1 -pyramidOnly -ps 2048 2048 -co\
     TILED=YES -co COMPRESS=JPEG -targetDir image_tiles " + tiff_file
    if os.path.exists("image_tiles"):
        os.system(command1)
    else:
        os.system(command0)
        os.system(command1)


def tile_directory(directory):

    for file in os.listdir(directory):
        tile_tiff(directory + "/" + file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(epilog='')
    parser.add_argument('tiff_file')

    args = parser.parse_args()
    tiff = args.tiff_file

    if os.path.isfile(tiff):
        tile_tiff(tiff)
    if os.path.isdir(tiff):
        tile_directory(tiff)

import pandas as pd
import json
import geopandas as gpd

tomnod = gpd.read_file("DG_final_v2.geojson")
tifRange = pd.read_csv("tifRange-tiles-run-1.csv", header=None,
                       names=['tif_id', 'minxy','maxxy'])

# catalog_id == complete_c

def process_tup(tup):
    return [float(ele) for ele in (tup.strip('()').split(','))]


for i in range(len(tifRange)):
    tifRange.at[i,'catalog_id'] = (tifRange.iloc[i]['tif_id'].split(sep = "_")[1])
    tifRange.at[i, 'minxy'] = process_tup(tifRange.iloc[i]["minxy"])
    tifRange.at[i, 'maxxy'] = process_tup(tifRange.iloc[i]["maxxy"])


for i in range(len(tomnod)):
    x = tomnod.iloc[i]['label']
    if x == "Flooded / Damaged Building":
        tomnod.at[i,'type_id'] = '1'
    elif x == "Flooded / Blocked Road":
        tomnod.at[i,'type_id'] = '2'
    elif x == "Trash Heap":
        tomnod.at[i,'type_id'] = '3'
    elif x == "Blocked Bridge":
        tomnod.at[i,'type_id'] = '4'


tomnod['tif_id'] = ""
bad_list = []
for index_tomnod, row_tomnod in tomnod.iterrows():
    if index_tomnod % 1000 == 0:
        print('tomnod row: ', index_tomnod)
    tifRange_temp = tifRange[tifRange.catalog_id == row_tomnod['complete_c']]
    for index_tif, row_tif in tifRange_temp.iterrows():
        if row_tif['minxy'][0] <= row_tomnod['geometry'].exterior.coords.xy[0][4] <= row_tif['maxxy'][0] \
            and row_tif['minxy'][1] <= row_tomnod['geometry'].exterior.coords.xy[1][4] <= row_tif['maxxy'][1]:
            if tomnod.at[index_tomnod, 'tif_id'] == "":
                tomnod.at[index_tomnod, 'tif_id'] = row_tif["tif_id"]
                print ('yaaas')
            elif tomnod.at[index_tomnod, 'tif_id'] != "":
                tomnod = tomnod.append(tomnod.iloc[index_tomnod], ignore_index=True)
                tomnod.at[index_tomnod, 'tif_id'] = row_tif["tif_id"]
        # else:
            # bad_list.append([row_tomnod.name, row_tomnod['geometry'], row_tif['minxy'], row_tif['maxxy'], row_tomnod['complete_c']])
    


tomnod_out = tomnod[['label', 'type_id', 'complete_c', 'tif_id', 'geometry']]
tomnod_out.columns = ["label", "TYPE_ID", "CAT_ID", "IMAGE_ID", 'geometry']
