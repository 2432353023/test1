import glob

def txt2bioes(txt_file):
    with open(txt_file,'r',encoding='utf-8') as f, open(
        'cluener' + txt_file.rstrip('txt') + 'bioes','w',encoding='utf-8') as fout:
        for line in f:
            if line in ['\n','\r\n']:
                fout.write('\n')
            awords = line.split(' ')
            if len(awords)==2 :
                words = awords[0]
                sig = awords[1]
                if len(words) == 1:
                    fout.write(words+" "+"S_"+sig[:-1]+"\n")
                elif len(words) == 2:
                    fout.write(words[0]+" "+"B_"+sig[:-1]+"\n")
                    fout.write(words[1]+" "+"E_"+sig[:-1]+"\n")
                elif len(words)>2:
                    fout.write(words[0] + " " + "B_" + sig[:-1] + "\n")
                    oword = words[1:-1]
                    for word in oword:
                        fout.write(word+ " " + "I_" + sig[:-1] + "\n")
                    fout.write(words[-1] + " " + "E_" + sig[:-1] + "\n")
def main():
    txt_files = glob.glob('*.txt')
    for file in txt_files:
        txt2bioes(file)

if __name__ == "__main__":
    main()