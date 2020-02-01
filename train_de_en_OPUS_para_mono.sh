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
  --max_epoch)
    MAX_EPOCH="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$SRC" != 'en' ]; then
    OTHER_LANG=$SRC
else
    OTHER_LANG=$TGT
fi
echo $OTHER_LANG

if [ "$SRC" \< "$TGT" ]; then
    ORDERED_SRC=$SRC
    ORDERED_TGT=$TGT
else
    ORDERED_SRC=$TGT
    ORDERED_TGT=$SRC
fi

epoch_size=$(cat /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT/train.$SRC | wc -l)
max_epoch_size=300000
epoch_size=$((epoch_size>max_epoch_size ?  max_epoch_size : epoch_size))
echo $epoch_size

python -W ignore train.py \
    --exp_name umt_para_mono_$DATA_NAME\_$SRC\_$TGT \
    --dump_path ./tmp/ \
    --reload_model /data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth,/data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth \
    --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$DATA_NAME/processed_para_mono/$ORDERED_SRC-$ORDERED_TGT \
    --lgs $SRC-$TGT \
    --ae_steps $SRC,$TGT \
    --bt_steps $SRC-$TGT-$SRC,$TGT-$SRC-$TGT \
    --word_shuffle 3 \
    --word_dropout 0.1 \
    --word_blank 0.1 \
    --lambda_ae '0:1,100000:0.1,300000:0' \
    --encoder_only false \
    --emb_dim 1024 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 1400 \
    --batch_size 32 \
    --bptt 256 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size $epoch_size \
    --eval_bleu true \
    --stopping_criterion valid_$SRC-$TGT\_mt_bleu,3 \
    --validation_metrics valid_$SRC-$TGT\_mt_bleu \
    --max_epoch $MAX_EPOCH \
    --max_len 150 \