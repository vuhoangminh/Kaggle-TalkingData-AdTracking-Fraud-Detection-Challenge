import gzip    

list_of_files = '../output_gz/*.gz'
for filename in list_of_files:
  with gzip.open(filename, 'rt') as f:
    data = f.read()
    with open(filename[:-3], 'wt') as f:
      f.write(data)