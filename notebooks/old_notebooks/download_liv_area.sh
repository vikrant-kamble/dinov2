
# Set the source S3 paths
SOURCE_PATH_1="s3://cape-data-east/datascience_storage/tldr-datasets/oblique_cubicasa_train_chips_pj_living_area_dev_v4_chunks/605c8eca823a42f8b981349971f2654c/*"
SOURCE_PATH_2="s3://cape-data-east/datascience_storage/tldr-datasets/oblique_cubicasa_val_chips_pj_living_area_dev_v4/4716083f83464d52b1f0a1000e39622a/*"

# Set the destination local folders
DEST_FOLDER_1="/cnvrg/oblique_cubicasa_train_chips_pj_living_area_dev_v4_chunks"
DEST_FOLDER_2="/cnvrg/oblique_cubicasa_val_chips_pj_living_area_dev_v4"

# Download files using s5cmd
s5cmd cp "$SOURCE_PATH_1" "$DEST_FOLDER_1" > /dev/null
echo "DONE: Downloaded files from $SOURCE_PATH_1 to $DEST_FOLDER_1"
s5cmd cp "$SOURCE_PATH_2" "$DEST_FOLDER_2" > /dev/null
echo "DONE: Downloaded files from $SOURCE_PATH_2 to $DEST_FOLDER_2"