import subprocess

with open('/content/test.txt', 'w') as f:
    f.write('hello world')

res = subprocess.run(["curl", "--upload-file", "/content/test.txt", "https://transfer.sh/test.txt"], capture_output=True, text=True)
print("transfer.sh stdout:", res.stdout)
print("transfer.sh stderr:", res.stderr)
