import glob

res=[]
for filename in glob.glob('*.py'):
    f = open(filename,'r',encoding="utf-8")
    res.append("#"+filename+"\n")
    res.append("```python3\n")
    res.extend(f.readlines())
    res.append("\n")
    res.append("```\n")
    f.close()
print(len(res))
print(res[0])
f=open("all.py","w",encoding="utf-8")
for line in res:
    f.write(line)
f.close()