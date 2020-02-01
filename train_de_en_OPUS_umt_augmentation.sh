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
  --src_data_name)
    SRC_DATA_NAME="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

epoch_size=$(cat /data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$DATA_NAME/umt_${SRC_DATA_NAME}_augmentation/train.$SRC | wc -l)
epoch_size=$((epoch_size/2))
echo $epoch_size

if [ $DATA_NAME == "KORAN" ]; then
    max_epoch=10
elif [ $DATA_NAME == "EMEA" ]; then
    max_epoch=10
elif [ $DATA_NAME == "IT" ]; then
    max_epoch=20
else
    max_epoch=15
fi
echo $max_epoch

python -W ignore train.py \
    --exp_name umt_src_${SRC_DATA_NAME}_augmentation_tgt_${DATA_NAME}_${SRC}_${TGT} \
    --dump_path ./tmp/ \
    --reload_model '/data/medg/misc/jindi/nlp/embeddings/XLM/mlm_ende_1024.pth,/data/medg/misc/jindi/nlp/embeddings/XLM/mlm_ende_1024.pth' \
    --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$DATA_NAME/umt_${SRC_DATA_NAME}_augmentation \
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
    --tokens_per_batch 1500 \
    --batch_size 32 \
    --bptt 256 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size $epoch_size \
    --eval_bleu true \
    --stopping_criterion valid_$SRC-$TGT\_mt_bleu,5 \
    --validation_metrics valid_$SRC-$TGT\_mt_bleu \
    --max_epoch $max_epoch \
    --max_len 150 \