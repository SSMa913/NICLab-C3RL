if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

### M

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --features M \
    --seq_len 672 \
    --label_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.4 \
    --lambda_pred 0.6 >logs/LongForecasting/informer'_'Ettm1_672'_'24.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.1 \
    --lambda_pred 0.9 >logs/LongForecasting/informer'_'Ettm1_96'_'48.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --features M \
    --seq_len 384 \
    --label_len 384 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.6 \
    --lambda_pred 0.4 >logs/LongForecasting/informer'_'Ettm1_384'_'96.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --features M \
    --seq_len 672 \
    --label_len 288 \
    --pred_len 288 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.1 \
    --lambda_pred 0.9 >logs/LongForecasting/informer'_'Ettm1_672'_'288.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --features M \
    --seq_len 672 \
    --label_len 384 \
    --pred_len 672 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.7 \
    --lambda_pred 0.3 >logs/LongForecasting/informer'_'Ettm1_672'_'672.log


