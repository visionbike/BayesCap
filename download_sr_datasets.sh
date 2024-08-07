FILE=$1

DATASETS=("SRGAN_ImageNet" "Set5" "Set14" "BSDS100")
i=0
for DATASET in "${DATASETS[@]}"; do
  if [ "$FILE" == "$DATASET" ]; then
    # download the selected dataset and move validation images to labeled subfolders
    URL="https://huggingface.co/datasets/goodfellowliu/$FILE/resolve/main/$FILE.zip"
    ZIP_FILE=./data/$FILE.zip
    mkdir -p ./data/"$FILE"
    wget -N $URL -O "$ZIP_FILE"
    unzip "$ZIP_FILE" -d ./data/"$FILE"
    rm "$ZIP_FILE"
    break
  else
    i=$((i + 1))
  fi
done

if [ "$i" == "${#DATASETS[@]}" ]; then
  echo "Available arguments are ${DATASETS[*]}"
  exit 1
fi