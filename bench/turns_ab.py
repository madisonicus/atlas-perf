import os,subprocess,time
for arm,flag in [("base","0"),("turn","1")]:
    env=dict(os.environ,OLLAMA_MLX_TURN_SNAPSHOTS=flag,OLLAMA_KEEP_ALIVE="-1",OLLAMA_CONTEXT_LENGTH="262144",OLLAMA_HOST="127.0.0.1:11581")
    pr=subprocess.Popen(["./ollama-prof","serve"],env=env,stdout=open(f"turns-{arm}.log","w"),stderr=subprocess.STDOUT)
    time.sleep(5)
    try:
        for turns in ["1","3","6","12","20"]:
            out=subprocess.run(["python3","/tmp/ttft.py","qwen3.8:27b-int4u"],env=dict(os.environ,TURNS=turns,TTFT_URL="http://127.0.0.1:11581/api/chat"),capture_output=True,text=True)
            print(arm,"TURNS",turns,out.stdout.strip().splitlines()[-1],flush=True)
    finally:
        pr.terminate(); pr.wait()
