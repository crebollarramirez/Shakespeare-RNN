import os

for config_file in os.listdir("configs"):
    config_path = os.path.join("configs", config_file)
    command = f"python3 main.py {config_path}"
    print(f"Executing command: {command}")
    os.system(command)
