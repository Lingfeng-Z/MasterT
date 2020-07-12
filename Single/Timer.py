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


        print("Phase 1 complete")
        sleep(10)

        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/Basis.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/Punc.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/POS.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/Spell.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/Stem.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/Stop.py")

        sleep(10)

        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Single/All_news.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/Dict_Create.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/Negation.py")
        os.system("python /home/lingfengzhang/Code/Sync/MasterThesis/Wiki_Single/All_news.py")
        return 0
run()