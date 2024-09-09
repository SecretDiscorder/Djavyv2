## Python script to print consequtive characters from unicode code
output = ''
for i in range(128,230):
	j = hex(i)
	k = str(j).replace("0x", "")

	code = 'A9' + k
	output = output + (unichr(int(code, 16))) + ' '

print (output)
