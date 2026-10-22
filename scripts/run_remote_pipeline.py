import runpy
import sys

print("=== Starting Remote NewsSeek Pipeline on A100 ===")
sys.stdout.flush()
runpy.run_path("/content/colab_run_newsseek_a100.py", run_name="__main__")
