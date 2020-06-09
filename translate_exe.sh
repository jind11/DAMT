#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  --data_name)
    DATA_NAME="$2"; shift 2;;
  --model_name)
    MODEL_NAME="$2"; shift 2;;
  --model_dir)
    MODEL_DIR="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$SRC" \< "$TGT" ]; then
    ORDERED_SRC=$SRC
    ORDERED_TGT=$TGT
else
    ORDERED_SRC=$TGT
    ORDERED_TGT=$SRC
fi

OUT_DIR=data/$ORDERED_SRC-$ORDERED_TGT/$DATA_NAME/back_translate/$MODEL_NAME
mkdir -p $OUT_DIR

python -W ignore translate.py \
    --exp_name $MODEL_NAME\_$SRC\_to_$TGT \
    --dump_path ./back_translate/ \
    --model_path $MODEL_DIR/best-valid_$SRC-$TGT\_mt_bleu.pth \
    --src_data_path data/$ORDERED_SRC-$ORDERED_TGT/$DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT/train.$ORDERED_SRC-$ORDERED_TGT.$SRC \
    --output_path_source $OUT_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$SRC \
    --output_path_target $OUT_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$TGT \
    --src_lang $SRC \
    --tgt_lang $TGT \
    --batch_size 128 \