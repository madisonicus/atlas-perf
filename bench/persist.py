import os,subprocess,time,json,urllib.request,sys
binp=sys.argv[1]; port=sys.argv[2]
env=dict(os.environ,OLLAMA_MLX_TURN_SNAPSHOTS="1",OLLAMA_KEEP_ALIVE="-1",OLLAMA_CONTEXT_LENGTH="65536",OLLAMA_HOST=f"127.0.0.1:{port}",OLLAMA_KV_CACHE_DIR="/tmp/kvc-"+port)
def srv(): 
    p=subprocess.Popen([binp,"serve"],env=env,cwd=os.path.dirname(binp),stdout=open(f"/tmp/persist-{port}.log","a"),stderr=subprocess.STDOUT); time.sleep(6); return p
import random
sys.path.insert(0,"/tmp")
random.seed(7)
hist=[{"role":"system","content":"You talk with Ben."}]
for t in range(6):
    q=f"Turn {t} [{random.getrandbits(40):x}]: "+" ".join(f"note{random.randrange(10**6)}" for _ in range(350))
    hist+=[{"role":"user","content":q},{"role":"assistant","content":"Noted. "+" ".join(f"ok{random.randrange(10**6)}" for _ in range(150))}]
M="qwen3.8:27b-int4u-dflash2"
def chat(msgs,model=M):
    b={"model":model,"stream":False,"think":False,"messages":msgs,"options":{"num_predict":30,"temperature":0}}
    r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/chat",json.dumps(b).encode()),timeout=900))
    return r, round((r.get("load_duration",0)+r["prompt_eval_duration"])/1e9,2), round(r["prompt_eval_duration"]/1e9,2)
p=srv()
try:
    r1,t1,_=chat(hist+[{"role":"user","content":"Summarize the last turn."}]); print("turn1 cold", t1)
    h2=hist+[{"role":"user","content":"Summarize the last turn."},{"role":"assistant","content":r1["message"]["content"]}]
    time.sleep(5)
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/generate",json.dumps({"model":M,"keep_alive":0}).encode())).read(); time.sleep(5)
    r2,t2,pe2=chat(h2+[{"role":"user","content":"And before it?"}]); print("after unload: first word",t2,"prompt part",pe2,"cached",r2.get("prompt_eval_cached_count",0),"of",r2["prompt_eval_count"])
    h3=h2+[{"role":"user","content":"And before it?"},{"role":"assistant","content":r2["message"]["content"]}]
    time.sleep(5)
    p.terminate(); p.wait(); time.sleep(3); p=srv()
    r3,t3,pe3=chat(h3+[{"role":"user","content":"Thanks."}]); print("after restart: first word",t3,"prompt part",pe3,"cached",r3.get("prompt_eval_cached_count",0),"of",r3["prompt_eval_count"])
finally:
    p.terminate(); p.wait()
