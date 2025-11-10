if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

### M

python -u main_informer.py \
    --model informer \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --data weather \
    --features M \
    --attn prob \
    --d_layers 2 \
    --e_layers 3 \
    --itr 1 \
    --label_len 168 \
    --pred_len 24 \
    --seq_len 168 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --lambda_sim 0.3 \
    --lambda_pred 0.7 >logs/LongForecasting/informer'_'weather_168'_'24.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --data weather \
    --features M \
    --attn prob \
    --d_layers 1 \
    --e_layers 2 \
    --itr 1 \
    --label_len 96 \
    --pred_len 48 \
    --seq_len 96 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --lambda_sim 0.9 \
    --lambda_pred 0.1 >logs/LongForecasting/informer'_'weather_96'_'48.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --data weather \
    --features M \
    --attn prob \
    --d_layers 2 \
    --e_layers 3 \
    --itr 1 \
    --label_len 168 \
    --pred_len 168 \
    --seq_len 336 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --lambda_sim 0.7 \
    --lambda_pred 0.3 >logs/LongForecasting/informer'_'weather_336'_'168.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --data weather \
    --features M \
    --attn prob \
    --d_layers 2 \
    --e_layers 3 \
    --itr 1 \
    --label_len 168 \
    --pred_len 336 \
    --seq_len 720 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --lambda_sim 0.8 \
    --lambda_pred 0.2 >logs/LongForecasting/informer'_'weather_720'_'336.log

python -u main_informer.py \
    --model informer \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --data weather \
    --features M \
    --attn prob \
    --d_layers 2 \
    --e_layers 3 \
    --itr 1 \
    --label_len 336 \
    --pred_len 720 \
    --seq_len 720 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --lambda_sim 0.7 \
    --lambda_pred 0.3 >logs/LongForecasting/informer'_'weather_720'_'720.log