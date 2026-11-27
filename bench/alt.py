import os,subprocess,time,urllib.request,json,re
SETS={
"prose":["Explain in about 300 words how ZFS snapshots work and why they are cheap.","Describe the history of the printing press in about 300 words.","Explain how vaccines train the immune system, in about 300 words.","Write about 300 words on why cities build ring roads.","Explain how a heat pump works in winter, about 300 words.","Describe the causes of the 1929 stock market crash in about 300 words."],
"chat":["My D&D group meets Saturday. Suggest three short side quests in Vallaki for level 5 characters.","I'm hosting six friends for dinner, one is vegan. Suggest a menu.","What should I pack for a 4-day hiking trip in October in Vermont?","Give me ideas for a 10-year-old's birthday party at home.","I keep waking up at 4am. Any tips?","Recommend five board games for two players and say why."],
"code":["Write a Python function that parses an nginx access log line into a dict, with a docstring and 5 pytest tests.","Write a Go HTTP handler that returns JSON health status with uptime.","Write a bash script that rotates log files older than 7 days with gzip.","Write a TypeScript debounce function with types and a short usage example.","Write a SQL query and schema for the top 5 customers by revenue per month.","Write a Rust function that reads a CSV file and sums one column."]}
for kind,prompts in SETS.items():
    env=dict(os.environ,OLLAMA_MLX_ALT_STATS="1",OLLAMA_KEEP_ALIVE="-1",OLLAMA_CONTEXT_LENGTH="65536",OLLAMA_HOST="127.0.0.1:11570")
    pr=subprocess.Popen(["./ollama-prof","serve"],env=env,stdout=open(f"alt-{kind}.log","w"),stderr=subprocess.STDOUT)
    time.sleep(5)
    for p in prompts:
        b={"model":"qwen3.8:27b-int4u","stream":False,"think":False,"messages":[{"role":"user","content":p}],"options":{"num_predict":400,"temperature":0}}
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11570/api/chat",json.dumps(b).encode()),timeout=900).read()
    pr.terminate(); pr.wait(); time.sleep(2)
    lines=re.findall(r'second choice at rejection" (.*)',open(f"alt-{kind}.log").read())
    acc=re.findall(r'acceptance=([0-9.]+) avg_draft=([0-9.]+) max_draft=\d+ avg_accepted=([0-9.]+)',open(f"alt-{kind}.log").read())
    print(kind, lines[-1] if lines else "none", "| per-request acc/avg_draft/avg_acc:", acc[1:])
