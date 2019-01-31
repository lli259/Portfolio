import os
allresult=[filename for filename in os.listdir('./ham_result_values') if filename.endswith(".csv")]

with open("algorithm_runs.csv","w") as fo:
	fo.write("algorithm,instance_id,runtime\n")

for f_ in allresult:
	with open("./ham_result_values/"+f_,"r") as f:
		lines=f.readlines()
		alldata=[line.split(",") for line in lines]
		
		#add penalty 10 time 0+
		#for line in alldata[1:]:
			#if "0+" in line[1]:
				#line[2]=str(float(line[2])*10)

		outdata=[]
		for line in alldata[1:]:
			chosenList=[f_.split("_")[0]]+[line[0]]+[line[2]]
			outdata.append(",".join(chosenList)+"\n")

		with open("algorithm_runs.csv","a") as fo: 
			for line in outdata:
				fo.write(line)		
	

