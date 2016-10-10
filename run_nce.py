from subprocess import call

N = 3

for i in range(N):
    call("python word2vec_nce.py nce"+ str(i) +" 1.0 models/data_unigram.dat", shell=True)


