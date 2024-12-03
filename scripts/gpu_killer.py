import subprocess

def kill_gpu_python_processes():
    try:
        # Run the fuser command to get processes using /dev/nvidia*
        result = subprocess.run('fuser -v /dev/nvidia*', shell=True, capture_output=True, text=True, check=True)
        
        # Output of fuser command
        output = result.stdout
        subprocess.run(f'kill -9 {output.replace("kernel", "")}', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
    except FileNotFoundError:
        print("The 'fuser' command is not found. Make sure it is installed and available in your PATH.")

# Execute the function
kill_gpu_python_processes()
