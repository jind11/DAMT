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

data_dir=/home/ubuntu/proj/data/$ORDERED_SRC-$ORDERED_TGT

epoch_size=$(cat ${data_dir}/processed/$ORDERED_SRC-$ORDERED_TGT/train.$ORDERED_SRC-$ORDERED_TGT.$SRC | wc -l)
max_epoch_size=400000
epoch_size=$((epoch_size>max_epoch_size ?  max_epoch_size : epoch_size))
echo $epoch_size

python -W ignore train.py \
    --exp_name sup_$SRC\_$TGT \
    --dump_path ./tmp/ \
    --data_path ${data_dir}/processed/$ORDERED_SRC-$ORDERED_TGT \
    --lgs $SRC-$TGT \
    --mt_steps $SRC-$TGT \
    --encoder_only false \
    --emb_dim 512 \
    --n_layers 5 \
    --n_heads 2 \
    --dropout 0.4 \
    --attention_dropout 0.2 \
    --gelu_activation true \
    --tokens_per_batch 12000 \
    --batch_size 256 \
    --bptt 256 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001 \
    --epoch_size $epoch_size \
    --eval_bleu true \
    --stopping_criterion valid_$SRC-$TGT\_mt_bleu,6 \
    --validation_metrics valid_$SRC-$TGT\_mt_bleu \
    --max_epoch 100 \
    --max_len 150 \
    --bpe_type sentencepiece \