import argparse
import os
import shutil
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--contest-name", required=True, type=str, help="contest name")
    parser.add_argument("-i", "--create-input-files", type=int, help="1 or 0", default=1)
    args = parser.parse_args()

    name = str(args.contest_name)
    if os.path.exists(name):
        op = input(f"Folder {name} exists, still want to create template files? [Y/n]")
        if 'n' in op:
            print("Bye!")
            exit(0)
    
    os.makedirs(name, exist_ok=True)
    letters = "abcdefgh"
    os.chdir(name)
    for letter in letters:
        shutil.copy('../template/a.cc', f'{letter}.cc')
        if args.create_input_files == 1:
            open(f'{letter}.in','w').close()
    for file in glob('../template/*'):
        print(file)
        shutil.copy(file, '.')

if __name__ == '__main__':
    main()
