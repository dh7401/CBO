import subprocess

for i in range(5):
    for problem in ["mopta", "lunar"]:
        f = open(f"{problem}_turbo_{i}.txt", "w")
        subprocess.Popen(["python", "turbo.py", "--seed", str(i), "--problem", problem], stdout=f, stderr=f).wait()
        f.close()

        for search in ["ets", "hts"]:
            f = open(f"{problem}_{search}_{i}.txt", "w")
            subprocess.Popen(["python", "mtgp.py", "--seed", str(i), "--problem", problem, "--search", search], stdout=f, stderr=f).wait()
            f.close()
