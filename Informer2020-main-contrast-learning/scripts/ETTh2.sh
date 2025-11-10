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
    --data_path ETTh2.csv \
    --data ETTh2 \
    --features M \
    --seq_len 48 \
    --label_len 48 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.2 \
    --lambda_pred 0.8 >logs/LongForecasting/informer'_'Etth2_48'_'24.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTh2.csv \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.9 \
    --lambda_pred 0.1 >logs/LongForecasting/informer'_'Etth2_96'_'48.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTh2.csv \
    --data ETTh2 \
    --features M \
    --seq_len 336 \
    --label_len 336 \
    --pred_len 168 \
    --e_layers 3 \
    --d_layers 2 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.8 \
    --lambda_pred 0.2 >logs/LongForecasting/informer'_'Etth2_336'_'168.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTh2.csv \
    --data ETTh2 \
    --features M \
    --seq_len 336 \
    --label_len 168 \
    --pred_len 336 \
    --e_layers 3 \
    --d_layers 2 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.6 \
    --lambda_pred 0.4 >logs/LongForecasting/informer'_'Etth2_336'_'336.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/ETT/ \
    --data_path ETTh2.csv \
    --data ETTh2 \
    --features M \
    --seq_len 720 \
    --label_len 336 \
    --pred_len 720 \
    --e_layers 3 \
    --d_layers 2 \
    --attn prob \
    --des 'Exp' \
    --itr 1 \
    --lambda_sim 0.9 \
    --lambda_pred 0.1 >logs/LongForecasting/informer'_'Etth2_720'_'720.log


