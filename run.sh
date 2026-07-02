python scripts/data_preparation/filter_rules.py $WIKIDATA_KG_ROOT/results/rules_with_nl_mid2.csv data/results/current/filtered_association_rules.csv --head-coverage 0.05 --std-confidence 0.5 --pca-confidence 0.9 --positive-examples 100

python scripts/data_preparation/split_csv.py data/processed/rules/rules_with_nl_mid2.csv data/processed/split_chunks
python scripts/analysis/analyze_logic.py --base-url https://api.deepseek.com/v1 --api-key <YOUR-API-KEY> --model deepseek-chat --input-dir data/processed/split_chunks --output-dir data/processed/analyzed_chunks

python scripts/analysis/filter_logic_y.py --input data/processed/rules/rules_with_nl_analyzed.csv --output data/processed/rules/rules_logic_y_filtered.csv --sep ',' --max_examples 1000

python scripts/preprocessing/match_rules_wikidata.py --memgraph_uri bolt://localhost:7687 --rules_file data/processed/rules/rules_logic_y_filtered.csv --output_dir data/examples/positive

python scripts/mutation/mutate_examples.py --memgraph_uri bolt://localhost:7687 --input_dir data/examples/positive --output_dir data/examples/mutated
python scripts/question_generation/construct_questions.py



python scripts/llm_interaction/llm_answer_and_extract.py --base-url http://localhost:8000/v1 --api-key <YOUR-API-KEY> --model Qwen2.5-7B-Instruct --input-dir data/examples/questions --output-dir data/examples/llm_answers_Qwen2.5-7B-Instruct
python scripts/llm_interaction/llm_answer_and_extract.py --base-url http://localhost:8316/v1 --api-key nju-claude-api-proxy-712! --model deepseek-v3 --input-dir data/examples/questions --output-dir data/examples/llm_answers_deepseek-v3 --max-concurrent 1 --resume
python scripts/llm_interaction/llm_answer_and_extract.py --base-url https://open.bigmodel.cn/api/anthropic --api-key 7d013f24fe054cbebba3c36f30177706.NOJevJkadnHjrqTv --model GLM-5-Turbo --protocol anthropic --input-dir data/examples/questions --output-dir data/examples/llm_answers_GLM-5-Turbo --max-concurrent 1 --resume
python scripts/llm_interaction/llm_answer_and_extract.py --base-url https://api.deepseek.com/anthropic --api-key <YOUR-API-KEY> --model deepseek-v4-flash --protocol anthropic --input-dir data/examples/questions --output-dir data/examples/llm_answers_deepseek-v4-flash --max-concurrent 1 --resume
python scripts/llm_interaction/llm_answer_and_extract.py --base-url <YOUR-BASE-URL> --api-key <YOUR-API-KEY> --model gpt-5.5 --protocol anthropic --input-dir data/examples/questions --output-dir data/examples/llm_answers_gpt-5.5 --max-concurrent 1 --resume








python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/llm_answers_deepseek-v3 --output_dir data/examples/consistency_results_nli_deepseek-v3
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/llm_answers_Qwen2.5-7B-Instruct --output_dir data/examples/consistency_results_nli_Qwen2.5-7B-Instruct
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/llm_answers_GLM-5-Turbo --output_dir data/examples/consistency_results_nli_GLM-5-Turbo
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/llm_answers_deepseek-v4-flash --output_dir data/examples/consistency_results_nli_deepseek-v4-flash
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/llm_answers_gpt-5.5 --output_dir data/examples/consistency_results_nli_gpt-5.5



python scripts/analysis/analyze_consistency.py --data-dir data/examples/consistency_results_Qwen2.5-7B-Instruct --output-dir data/analysis/consistency_results_Qwen2.5-7B-Instruct
python scripts/analysis/summarize_consistency_stats.py --input data/examples/consistency_results_Qwen2.5-7B-Instruct --output consistency_stats_summary_Qwen2.5-7B-Instruct.csv
python scripts/analysis/summarize_by_question_type.py --input data/examples/consistency_results_Qwen2.5-7B-Instruct --output consistency_by_question_type_Qwen2.5-7B-Instruct.csv


python scripts/rqs/rq4_golden_dataset_sampling.py
python scripts/rqs/rq4_qaasker_mutations.py
python scripts/rqs/rq4_qaasker_llm_answer.py --base-url http://localhost:8000/v1 --api-key <YOUR-API-KEY> --model Qwen2.5-7B-Instruct
python scripts/rqs/rq4_qaasker_llm_answer.py --base-url https://api.siliconflow.cn/v1 --api-key <YOUR-API-KEY> --model deepseek-ai/DeepSeek-V3.2 --model-name deepseek-v3
python scripts/rqs/rq4_qaasker_llm_answer.py --base-url https://api.deepseek.com/anthropic --api-key <YOUR-API-KEY> --model deepseek-v4-flash --protocol anthropic --skip-processed
python scripts/rqs/rq4_qaasker_llm_answer.py --base-url https://open.bigmodel.cn/api/anthropic --api-key 7d013f24fe054cbebba3c36f30177706.NOJevJkadnHjrqTv --model GLM-5-Turbo --protocol anthropic --skip-processed
python scripts/rqs/rq4_qaasker_llm_answer.py --base-url <YOUR-BASE-URL> --api-key <YOUR-API-KEY> --model gpt-5.5 --protocol anthropic --skip-processed

python scripts/rqs/rq4_qaasker_evaluation.py --input-dir data/examples/golden_dataset_qaasker_answer_Qwen2.5_7B_Instruct
python scripts/rqs/rq4_qaasker_evaluation.py --input-dir data/examples/golden_dataset_qaasker_answer_deepseek_v3
python scripts/rqs/rq4_qaasker_evaluation.py --input-dir data/examples/golden_dataset_qaasker_answer_deepseek_v4_flash
python scripts/rqs/rq4_qaasker_evaluation.py --input-dir data/examples/golden_dataset_qaasker_answer_GLM_5_Turbo
python scripts/rqs/rq4_qaasker_evaluation.py --input-dir data/examples/golden_dataset_qaasker_answer_gpt_5.5

python scripts/rqs/rq2_inconsistency_overview.py

python scripts/rqs/rq4_qaqa_mutations.py
python scripts/rqs/rq4_qaqa_llm_answer.py --base-url http://localhost:8000/v1 --api-key <YOUR-API-KEY> --model Qwen2.5-7B-Instruct
python scripts/rqs/rq4_qaqa_llm_answer.py --base-url https://api.siliconflow.cn/v1 --api-key <YOUR-API-KEY> --model deepseek-ai/DeepSeek-V3.2 --model-name deepseek-v3
python scripts/rqs/rq4_qaqa_llm_answer.py --base-url https://api.deepseek.com/anthropic --api-key <YOUR-API-KEY> --model deepseek-v4-flash --protocol anthropic --skip-processed
python scripts/rqs/rq4_qaqa_llm_answer.py --base-url https://open.bigmodel.cn/api/anthropic --api-key 7d013f24fe054cbebba3c36f30177706.NOJevJkadnHjrqTv --model GLM-5-Turbo --protocol anthropic --skip-processed
python scripts/rqs/rq4_qaqa_llm_answer.py --base-url <YOUR-BASE-URL> --api-key <YOUR-API-KEY> --model gpt-5.5 --protocol anthropic --skip-processed

python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_Qwen2.5_7B_Instruct
python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_deepseek_v3
python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_deepseek_v4_flash
python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_GLM_5_Turbo
python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_gpt_5.5




python scripts/rqs/rq4_drhall_mutations.py

python scripts/rqs/rq4_drhall_llm_answer.py --base-url http://localhost:8000/v1 --api-key <YOUR-API-KEY> --model Qwen2.5-7B-Instruct
python scripts/rqs/rq4_drhall_llm_answer.py --base-url https://api.siliconflow.cn/v1 --api-key <YOUR-API-KEY> --model deepseek-ai/DeepSeek-V3.2 --model-name deepseek-v3 --skip-processed
python scripts/rqs/rq4_drhall_llm_answer.py --base-url https://api.deepseek.com/anthropic --api-key <YOUR-API-KEY> --model deepseek-v4-flash --protocol anthropic --skip-processed
python scripts/rqs/rq4_drhall_llm_answer.py --base-url https://open.bigmodel.cn/api/anthropic --api-key 7d013f24fe054cbebba3c36f30177706.NOJevJkadnHjrqTv --model GLM-5-Turbo --protocol anthropic --skip-processed
python scripts/rqs/rq4_drhall_llm_answer.py --base-url <YOUR-BASE-URL> --api-key <YOUR-API-KEY> --model gpt-5.5 --protocol anthropic --skip-processed

python scripts/rqs/rq4_drhall_evaluation.py --input-dir data/examples/golden_dataset_drhall_answer_Qwen2.5_7B_Instruct
python scripts/rqs/rq4_drhall_evaluation.py --input-dir data/examples/golden_dataset_drhall_answer_deepseek_v3
python scripts/rqs/rq4_drhall_evaluation.py --input-dir data/examples/golden_dataset_drhall_answer_deepseek_v4_flash
python scripts/rqs/rq4_drhall_evaluation.py --input-dir data/examples/golden_dataset_drhall_answer_GLM_5_Turbo
python scripts/rqs/rq4_drhall_evaluation.py --input-dir data/examples/golden_dataset_drhall_answer_gpt_5.5




python scripts/rqs/rq4_kontest_mutations.py

python scripts/rqs/rq4_kontest_llm_answer.py --base-url http://localhost:8000/v1 --api-key <YOUR-API-KEY> --model Qwen2.5-7B-Instruct
python scripts/rqs/rq4_kontest_llm_answer.py --base-url https://api.siliconflow.cn/v1 --api-key <YOUR-API-KEY> --model deepseek-ai/DeepSeek-V3.2 --model-name deepseek-v3 --skip-processed
python scripts/rqs/rq4_kontest_llm_answer.py --base-url https://api.deepseek.com/anthropic --api-key <YOUR-API-KEY> --model deepseek-v4-flash --protocol anthropic --skip-processed
python scripts/rqs/rq4_kontest_llm_answer.py --base-url https://open.bigmodel.cn/api/anthropic --api-key 7d013f24fe054cbebba3c36f30177706.NOJevJkadnHjrqTv --model GLM-5-Turbo --protocol anthropic --skip-processed
python scripts/rqs/rq4_kontest_llm_answer.py --base-url <YOUR-BASE-URL> --api-key <YOUR-API-KEY> --model gpt-5.5 --protocol anthropic --skip-processed

python scripts/rqs/rq4_kontest_evaluation.py --input-dir data/examples/golden_dataset_kontest_answer_Qwen2.5_7B_Instruct
python scripts/rqs/rq4_kontest_evaluation.py --input-dir data/examples/golden_dataset_kontest_answer_deepseek_v3
python scripts/rqs/rq4_kontest_evaluation.py --input-dir data/examples/golden_dataset_kontest_answer_deepseek_v4_flash
python scripts/rqs/rq4_kontest_evaluation.py --input-dir data/examples/golden_dataset_kontest_answer_GLM_5_Turbo
python scripts/rqs/rq4_kontest_evaluation.py --input-dir data/examples/golden_dataset_kontest_answer_gpt_5.5




python scripts/rqs/rq4_metaqa_mutations.py --base-url http://localhost:8000/v1 --api-key <YOUR-API-KEY> --model Qwen2.5-7B-Instruct
python scripts/rqs/rq4_metaqa_mutations.py --base-url https://api.siliconflow.cn/v1 --api-key <YOUR-API-KEY> --model deepseek-ai/DeepSeek-V3.2 --model-name deepseek-v3 --skip-processed
python scripts/rqs/rq4_metaqa_mutations.py --base-url https://api.deepseek.com/anthropic --api-key <YOUR-API-KEY> --model deepseek-v4-flash --protocol anthropic --skip-processed
python scripts/rqs/rq4_metaqa_mutations.py --base-url https://open.bigmodel.cn/api/anthropic --api-key 7d013f24fe054cbebba3c36f30177706.NOJevJkadnHjrqTv --model GLM-5-Turbo --protocol anthropic --skip-processed
python scripts/rqs/rq4_metaqa_mutations.py --base-url <YOUR-BASE-URL> --api-key <YOUR-API-KEY> --model gpt-5.5 --protocol anthropic --skip-processed

python scripts/rqs/rq4_metaqa_evaluation.py --input-dir data/examples/golden_dataset_metaqa_answer_Qwen2.5_7B_Instruct
python scripts/rqs/rq4_metaqa_evaluation.py --input-dir data/examples/golden_dataset_metaqa_answer_deepseek_v3
python scripts/rqs/rq4_metaqa_evaluation.py --input-dir data/examples/golden_dataset_metaqa_answer_deepseek_v4_flash
python scripts/rqs/rq4_metaqa_evaluation.py --input-dir data/examples/golden_dataset_metaqa_answer_GLM_5_Turbo
python scripts/rqs/rq4_metaqa_evaluation.py --input-dir data/examples/golden_dataset_metaqa_answer_gpt_5.5


python scripts/llm_interaction/llm_answer_and_extract.py --base-url http://localhost:8000/v1 --api-key <YOUR-API-KEY> --model Qwen2.5-7B-Instruct --input-dir data/examples/golden_dataset --output-dir data/examples/golden_dataset_llm_answers_Qwen2.5-7B-Instruct
python scripts/llm_interaction/llm_answer_and_extract.py --base-url https://api.siliconflow.cn/v1 --api-key <YOUR-API-KEY> --model deepseek-ai/DeepSeek-V3.2 --input-dir data/examples/golden_dataset --output-dir data/examples/golden_dataset_llm_answers_deepseek-v3 --max-concurrent 1 --resume
python scripts/llm_interaction/llm_answer_and_extract.py --base-url https://api.deepseek.com/anthropic --api-key <YOUR-API-KEY> --model deepseek-v4-flash --protocol anthropic --input-dir data/examples/golden_dataset --output-dir data/examples/golden_dataset_llm_answers_deepseek-v4-flash --max-concurrent 1 --resume
python scripts/llm_interaction/llm_answer_and_extract.py --base-url https://open.bigmodel.cn/api/anthropic --api-key 7d013f24fe054cbebba3c36f30177706.NOJevJkadnHjrqTv --model GLM-5-Turbo --protocol anthropic --input-dir data/examples/golden_dataset --output-dir data/examples/golden_dataset_llm_answers_GLM-5-Turbo --max-concurrent 1 --resume
python scripts/llm_interaction/llm_answer_and_extract.py --base-url <YOUR-BASE-URL> --api-key <YOUR-API-KEY> --model gpt-5.5 --protocol anthropic --input-dir data/examples/golden_dataset --output-dir data/examples/golden_dataset_llm_answers_gpt-5.5 --max-concurrent 1 --resume




python scripts/rqs/rq5_dataset_splitter.py


python scripts/rqs/rq5_sharedgpt_converter.py --input data/examples/rq5_results/splits/train/train.csv
python scripts/rqs/rq5_sharedgpt_converter.py --input data/examples/rq5_results/splits/valid/valid.csv


python scripts/llm_interaction/llm_answer_and_extract.py --base-url http://localhost:8000/v1 --api-key <YOUR-API-KEY> --model Qwen2.5-7B-Instruct --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/baseline_answers_Qwen2.5-7B-Instruct

python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/baseline_answers_Qwen2.5-7B-Instruct --output_dir data/examples/rq5_results/baseline_consistency_Qwen2.5-7B-Instruct

python scripts/analysis/analyze_consistency.py \
  --data-dir data/examples/rq5_results/baseline_consistency_Qwen2.5-7B-Instruct \
  --output-dir data/analysis/rq5_baseline_consistency
python scripts/analysis/summarize_consistency_stats.py \
  --input data/examples/rq5_results/baseline_consistency_Qwen2.5-7B-Instruct \
  --output data/examples/rq5_results/baseline_consistency_summary.csv


python scripts/rqs/rq5_finetuning.py --train-file data/examples/rq5_results/splits/train/train_sharedgpt.jsonl --llamafactory-path $LLAMAFACTORY_PATH
python scripts/rqs/rq5_finetuning.py --train-file data/examples/rq5_results/splits/train/train_sharedgpt.jsonl --llamafactory-path $LLAMAFACTORY_PATH --batch-size 2 --gradient-accumulation 4 --learning-rate 2e-5 --cutoff-len 4096 --save-steps 10 --gpus 4,5



python scripts/llm_interaction/local_llm_answer_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --lora-adapter data/examples/rq5_results/lora_adapter/checkpoint-10 --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/finetuned_answers_checkpoint-10 --device cuda
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/finetuned_answers_checkpoint-10 --output_dir data/examples/rq5_results/finetuned_consistency_checkpoint-10

python scripts/llm_interaction/local_llm_answer_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --lora-adapter data/examples/rq5_results/lora_adapter/checkpoint-20 --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/finetuned_answers_checkpoint-20 --device cuda
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/finetuned_answers_checkpoint-20 --output_dir data/examples/rq5_results/finetuned_consistency_checkpoint-20

python scripts/llm_interaction/local_llm_answer_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --lora-adapter data/examples/rq5_results/lora_adapter/checkpoint-30 --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/finetuned_answers_checkpoint-30 --device cuda
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/finetuned_answers_checkpoint-30 --output_dir data/examples/rq5_results/finetuned_consistency_checkpoint-30

python scripts/llm_interaction/local_llm_answer_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --lora-adapter data/examples/rq5_results/lora_adapter/checkpoint-40 --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/finetuned_answers_checkpoint-40 --device cuda
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/finetuned_answers_checkpoint-40 --output_dir data/examples/rq5_results/finetuned_consistency_checkpoint-40

python scripts/llm_interaction/local_llm_answer_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --lora-adapter data/examples/rq5_results/lora_adapter/checkpoint-50 --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/finetuned_answers_checkpoint-50 --device cuda
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/finetuned_answers_checkpoint-50 --output_dir data/examples/rq5_results/finetuned_consistency_checkpoint-50

python scripts/llm_interaction/local_llm_answer_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --lora-adapter data/examples/rq5_results/lora_adapter/checkpoint-60 --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/finetuned_answers_checkpoint-60 --device cuda
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/finetuned_answers_checkpoint-60 --output_dir data/examples/rq5_results/finetuned_consistency_checkpoint-60

python scripts/llm_interaction/local_llm_answer_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --lora-adapter data/examples/rq5_results/lora_adapter/checkpoint-70 --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/finetuned_answers_checkpoint-70 --device cuda
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/finetuned_answers_checkpoint-70 --output_dir data/examples/rq5_results/finetuned_consistency_checkpoint-70

python scripts/llm_interaction/local_llm_answer_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --lora-adapter data/examples/rq5_results/lora_adapter/checkpoint-80 --input-dir data/examples/rq5_results/splits/test --output-dir data/examples/rq5_results/finetuned_answers_checkpoint-80 --device cuda
python scripts/analysis/qa_consistency_nli.py --preset accurate --answers_dir data/examples/rq5_results/finetuned_answers_checkpoint-80 --output_dir data/examples/rq5_results/finetuned_consistency_checkpoint-80

python scripts/analysis/analyze_consistency.py \
  --data-dir data/examples/rq5_results/finetuned_consistency_checkpoint-200 \
  --output-dir data/analysis/rq5_finetuned_consistency
python scripts/analysis/summarize_consistency_stats.py \
  --input data/examples/rq5_results/finetuned_consistency_checkpoint-200 \
  --output data/examples/rq5_results/finetuned_consistency_summary.csv



python scripts/rqs/rq5_plot_training_loss_pdf.py --log-dir data/examples/rq5_results/lora_adapter
