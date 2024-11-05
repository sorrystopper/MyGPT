model_path=/home/skw/MyGPT/ckpt/epoch1_batch_1599
output_path=/home/skw/MyGPT/ceval_output
vocab_path=/home/skw/MyGPT/model/vocab.txt

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
    --vocab_path ${vocab_path} \
    --device 0