import os,subprocess,time,urllib.request,json,statistics,re,sys
P={"prose":"Explain in about 300 words how ZFS snapshots work and why they are cheap.",
"code":"Write a Python function that parses an nginx access log line into a dict, with a docstring and 5 pytest tests.",
"chat":"My D&D group meets Saturday. Suggest three short side quests in Vallaki for level 5 characters."}
spec=json.loads(sys.argv[1]); RUNS=int(sys.argv[2]) if len(sys.argv)>2 else 6
# spec: {name: {"port":..,"model":..,"env":{..}}}
procs=[]
for name,a in spec.items():
    env=dict(os.environ,OLLAMA_KEEP_ALIVE="-1",OLLAMA_CONTEXT_LENGTH="65536",OLLAMA_HOST=f"127.0.0.1:{a['port']}",**a.get("env",{}))
    procs.append(subprocess.Popen([a.get("bin","./ollama-prof"),"serve"],env=env,stdout=open(f"arm-{name}.log","w"),stderr=subprocess.STDOUT))
time.sleep(5)
def gen(a,p):
    b={"model":a["model"],"stream":False,"think":False,"messages":[{"role":"user","content":p}],"options":{"num_predict":400,"temperature":0}}
    r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a['port']}/api/chat",json.dumps(b).encode()),timeout=900))
    return r["eval_count"]/(r["eval_duration"]/1e9)
res={}
arms=list(spec.items())
for name,a in arms: gen(a,"Say OK.")
try:
  for i in range(RUNS):
    for k,p in P.items():
        for name,a in (arms if i%2==0 else arms[::-1]):
            t=gen(a,p); res.setdefault((k,name),[]).append(t)
            with open("arms-results.jsonl","a") as fh: fh.write(json.dumps({"at":time.strftime("%FT%T"),"arm":name,"model":a["model"],"env":a.get("env",{}),"kind":k,"tok_s":round(t,1)})+"\n")
finally:
  for p in procs: p.terminate()
base=arms[0][0]; avg={}
for k in P:
    print(k.ljust(6)," ".join(f"{n}={statistics.median(res[(k,n)]):.1f}({statistics.median(res[(k,n)])/statistics.median(res[(k,base)])-1:+.1%})" for n,_ in arms))
for n,_ in arms:
    avg[n]=statistics.mean(statistics.median(res[(k,n)]) for k in P)
print("avg   "," ".join(f"{n}={v:.1f}({v/avg[base]-1:+.1%})" for n,v in avg.items()))
