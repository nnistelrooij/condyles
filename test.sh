export IN_DIR=test
export OUT_DIR=test/out

chmod 777 $OUT_DIR

docker load < condyles_segmentation.tar.gz
docker run \
    -v $IN_DIR:/input \
    -v $OUT_DIR:/output \
    condyles:segmentation
