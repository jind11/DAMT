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
  --src_data_name)
    SRC_DATA_NAME="$2"; shift 2;;
  --tgt_data_name)
    TGT_DATA_NAME="$2"; shift 2;;
  --remove_part)
    REMOVE_PART="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

TOKENS_PER_BATCH=700

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

epoch_size=$(cat /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT/train.$ORDERED_SRC-$ORDERED_TGT.$SRC | wc -l)
max_epoch_size=300000
epoch_size=$((epoch_size>max_epoch_size ?  max_epoch_size : epoch_size))
echo $epoch_size

if [ "$REMOVE_PART" == "pretrain" ]; then
    python -W ignore train.py \
        --exp_name semi_sup_back_ablation_remove_${REMOVE_PART}_src_$SRC_DATA_NAME\_tgt_$TGT_DATA_NAME\_$SRC\_$TGT \
        --dump_path ./tmp/ \
        --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT \
        --para_data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/sup_back_translate/$SRC_DATA_NAME \
        --lgs $SRC-$TGT \
        --ae_steps $SRC,$TGT \
        --bt_steps $SRC-$TGT-$SRC,$TGT-$SRC-$TGT \
        --mt_steps $SRC-$TGT \
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
        --tokens_per_batch $TOKENS_PER_BATCH \
        --batch_size 32 \
        --bptt 256 \
        --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
        --epoch_size $epoch_size \
        --eval_bleu true \
        --stopping_criterion valid_$SRC-$TGT\_mt_bleu,3 \
        --validation_metrics valid_$SRC-$TGT\_mt_bleu \
        --max_epoch 50 \
        --max_len 150
elif [ "$REMOVE_PART" == "bt" ]; then
    python -W ignore train.py \
        --exp_name semi_sup_back_ablation_remove_${REMOVE_PART}_src_$SRC_DATA_NAME\_tgt_$TGT_DATA_NAME\_$SRC\_$TGT \
        --dump_path ./tmp/ \
        --reload_model /data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth,/data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth \
        --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT \
        --para_data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/sup_back_translate/$SRC_DATA_NAME \
        --lgs $SRC-$TGT \
        --ae_steps $SRC,$TGT \
        --mt_steps $SRC-$TGT \
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
        --tokens_per_batch $TOKENS_PER_BATCH \
        --batch_size 32 \
        --bptt 256 \
        --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
        --epoch_size $epoch_size \
        --eval_bleu true \
        --stopping_criterion valid_$SRC-$TGT\_mt_bleu,3 \
        --validation_metrics valid_$SRC-$TGT\_mt_bleu \
        --max_epoch 50 \
        --max_len 150
elif [ "$REMOVE_PART" == "ae" ]; then
    python -W ignore train.py \
        --exp_name semi_sup_back_ablation_remove_${REMOVE_PART}_src_$SRC_DATA_NAME\_tgt_$TGT_DATA_NAME\_$SRC\_$TGT \
        --dump_path ./tmp/ \
        --reload_model /data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth,/data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth \
        --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT \
        --para_data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/sup_back_translate/$SRC_DATA_NAME \
        --lgs $SRC-$TGT \
        --bt_steps $SRC-$TGT-$SRC,$TGT-$SRC-$TGT \
        --mt_steps $SRC-$TGT \
        --lambda_ae '0:1,100000:0.1,300000:0' \
        --encoder_only false \
        --emb_dim 1024 \
        --n_layers 6 \
        --n_heads 8 \
        --dropout 0.1 \
        --attention_dropout 0.1 \
        --gelu_activation true \
        --tokens_per_batch $TOKENS_PER_BATCH \
        --batch_size 32 \
        --bptt 256 \
        --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
        --epoch_size $epoch_size \
        --eval_bleu true \
        --stopping_criterion valid_$SRC-$TGT\_mt_bleu,3 \
        --validation_metrics valid_$SRC-$TGT\_mt_bleu \
        --max_epoch 50 \
        --max_len 150
elif [ "$REMOVE_PART" == "ae-half" ]; then
    python -W ignore train.py \
        --exp_name semi_sup_back_ablation_remove_${REMOVE_PART}_src_$SRC_DATA_NAME\_tgt_$TGT_DATA_NAME\_$SRC\_$TGT \
        --dump_path ./tmp/ \
        --reload_model /data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth,/data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth \
        --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT \
        --para_data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/sup_back_translate/$SRC_DATA_NAME \
        --lgs $SRC-$TGT \
        --bt_steps $SRC-$TGT-$SRC,$TGT-$SRC-$TGT \
        --ae_steps $TGT \
        --mt_steps $SRC-$TGT \
        --lambda_ae '0:1,100000:0.1,300000:0' \
        --encoder_only false \
        --emb_dim 1024 \
        --n_layers 6 \
        --n_heads 8 \
        --dropout 0.1 \
        --attention_dropout 0.1 \
        --gelu_activation true \
        --tokens_per_batch $TOKENS_PER_BATCH \
        --batch_size 32 \
        --bptt 256 \
        --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
        --epoch_size $epoch_size \
        --eval_bleu true \
        --stopping_criterion valid_$SRC-$TGT\_mt_bleu,3 \
        --validation_metrics valid_$SRC-$TGT\_mt_bleu \
        --max_epoch 50 \
        --max_len 150
elif [ "$REMOVE_PART" == "bt,ae" ]; then
    python -W ignore train.py \
        --exp_name semi_sup_back_ablation_remove_${REMOVE_PART}_src_$SRC_DATA_NAME\_tgt_$TGT_DATA_NAME\_$SRC\_$TGT \
        --dump_path ./tmp/ \
        --reload_model /data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth,/data/medg/misc/jindi/nlp/embeddings/XLM/mlm_en${OTHER_LANG}_1024.pth \
        --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT \
        --para_data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME/sup_back_translate/$SRC_DATA_NAME \
        --lgs $SRC-$TGT \
        --ae_steps $TGT \
        --mt_steps $SRC-$TGT \
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
        --tokens_per_batch $TOKENS_PER_BATCH \
        --batch_size 32 \
        --bptt 256 \
        --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
        --epoch_size $epoch_size \
        --eval_bleu true \
        --stopping_criterion valid_$SRC-$TGT\_mt_bleu,3 \
        --validation_metrics valid_$SRC-$TGT\_mt_bleu \
        --max_epoch 50 \
        --max_len 150
fi


