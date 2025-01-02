import subprocess
import time

# 定义训练命令
train_command = [
    "CUDA_VISIBLE_DEVICES=0", "python3", "aste_train_prompt.py",
    "--gpus=1",
    "--precision=16",
    "--data_dir", "../data/ASTE-V2/14lap",
    "--model_name_or_path", "bert-base-uncased",
    "--output_dir", "../output/ASTE/14lap/",
    "--learning_rate", "3e-5",
    "--train_batch_size", "4",
    "--eval_batch_size", "1",
    "--warmup_steps", "100",
    "--lr_scheduler", "linear",
    "--gradient_clip_val", "1",
    "--weight_decay", "0.01",
    "--max_seq_length", "-1",
    "--max_epochs", "20",
    "--cuda_ids", "0",
    "--do_train",
    "--table_encoder", "resnet",
    "--num_table_layers", "2",
    "--span_pruning", "0.3",
    "--seq2mat", "tensorcontext",
    "--num_d", "64"
]

# 定义 seed 列表
seeds = [40, 50, 60, 70, 80]  # 5 个不同的 seed

# 循环训练
for seed in seeds:
    print(f"Starting training with seed={seed}")
    
    # 添加 seed 参数到训练命令
    command = train_command + ["--seed", str(seed)]
    
    # 启动训练进程
    process = subprocess.Popen(" ".join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 等待训练完成
    stdout, stderr = process.communicate()
    
    # 打印训练输出
    if stdout:
        print("Training output:", stdout.decode())
    if stderr:
        print("Training error:", stderr.decode())
    
    # 检查训练是否成功
    if process.returncode == 0:
        print(f"Training with seed={seed} completed successfully.")
    else:
        print(f"Training with seed={seed} failed.")
    
    # 等待一段时间（可选）
    time.sleep(1)

print("All training runs completed.")