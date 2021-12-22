import re

with open("/data/java/java_lang.bnf", 'r') as file:
    filedata = file.read()

filedata = re.sub(r"<([a-z_]+)>", r'\1', filedata)
filedata = filedata.replace("::=", ":")
filedata = re.sub(r'(\s)([a-z_]+)\s+:\s+""\s+\|\s+([a-z_]+)', r'\1\2 : \3?', filedata)

with open('/data/java/java_lang.lark', 'w') as file:
    file.write(filedata)