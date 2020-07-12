import os
from time import sleep


def run():
    while 1:
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/Basis.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/Punc.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/POS.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/Spell.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/Stem.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/Stop.py")
        #os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/Negation.py")
        #os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/All_news.py")

        print("Phase 1 complete")


        return 0
run()