from collections import defaultdict
from glob import glob
import sys
import re

glob_files = sys.argv[1]
loc_outfile = sys.argv[2]

def kaggle_bag(glob_files, loc_outfile, method="average", weights="weighted"):
  pattern = re.compile(r"(.)*_[w|W](\d*)_[.]*")    
  if method == "average":
    scores = defaultdict(float)
  with open(loc_outfile,"w") as outfile:
    #weight_list may be usefull using a different method
    weight_list = [1]*len(glob(glob_files))      
    # print (weight_list)
    for i, glob_file in enumerate( glob(glob_files) ):
      print("parsing: {}".format(glob_file))
      # sort glob_file by first column, ignoring the first line
      if weights == "weighted":
         weight = pattern.match(glob_file)
         if weight and weight.group(2):
            print("Using weight: {}".format(weight.group(2)))
            weight_list[i] = weight_list[i]*int(weight.group(2))
         else:
            print("Using weight: 1")
    #   print (weight_list)            
      lines = open(glob_file).readlines()
    #   print('lines:',lines)
    
      for e, line in enumerate( lines ):
        if i == 0 and e == 0:
          outfile.write(line)
        if e > 0:
          row = line.strip().split(",")
          scores[(e,row[0])] += float(row[1])*weight_list[i]
        #   print('row:', row)
        #   print('scores e row 0:',scores[(e,row[0])])
        #   print('row 1:',row[1])
        #   print('scores:',scores)
    for j,k in sorted(scores):
      outfile.write("%s,%f\n"%(k,scores[(j,k)]/(sum(weight_list))))
    print("wrote to {}".format(loc_outfile))

kaggle_bag(glob_files, loc_outfile)