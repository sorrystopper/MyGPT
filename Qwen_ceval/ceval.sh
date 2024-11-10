model_path=/home/xli/models/Qwen2___5-1___5B-Instruct
output_path=/home/xli/skw/MyGPT/Qwen_ceval/ceval_output

python eval.py \
    --model_path ${model_path} \
    --cot False \
    --few_shot False \
    --with_prompt True \
    --constrained_decoding True \
    --temperature 0.9 \
    --n_times 1 \
    --ntrain 5 \
    --do_save_csv False \
    --do_test False \
    --output_dir ${output_path} \
    --device 1