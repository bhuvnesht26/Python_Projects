from itertools import product
import time
from tqdm import tqdm
import os
import shutil
import re


class patternFinder:
    def __init__(self, w_directory, k_mer_val):
        self.working_directory = w_directory
        self.k_mer_value = k_mer_val
        self.patterns = self.generate_patterns()
        self.pattern_freq = {}
        for filename in os.listdir(self.working_directory):
            if filename.endswith(".fa") or filename.endswith(".fasta"):
                t1 = time.time()
                input_file = os.path.join(self.working_directory, filename)
                sequence_list = self.read_file(input_file)
                binary_seq_list = self.create_binary(sequence_list)
                ID_seq_list = list(
                    zip(
                        [
                            binary_seq_list[index]
                            for index in range(len(binary_seq_list))
                            if index % 2 == 0
                        ],
                        [
                            binary_seq_list[index]
                            for index in range(len(binary_seq_list))
                            if index % 2 != 0
                        ],
                    )
                )
                pattern_result = self.find_pattern(ID_seq_list)
                for element in pattern_result:
                    self.pattern_freq[element.split(" ")[0]] = int(
                        element.split(" ")[1]
                    )
                sorted_pat_freq = dict(
                    sorted(
                        self.pattern_freq.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                with open(
                    f'{input_file.split(".fa")[0]}_frequency.txt', "w"
                ) as writefrequencyfile:
                    writefrequencyfile.writelines(
                        "{} - {} \n".format(Pattern, Frequency)
                        for Pattern, Frequency in sorted_pat_freq.items()
                    )
                t2 = time.time()
                print(f"It took {t2 - t1} seconds to process.")
                path = os.path.join(self.working_directory, input_file.split(".fa")[0])
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.move(input_file, path)
                shutil.move(f'{input_file.split(".fa")[0]}_frequency.txt', path)
            else:
                continue

    # removing characters other than 'A', 'T', 'G' and 'C'
    def read_file(self, fasta_file):
        seq_fasta = []
        with open(fasta_file) as readfile:
            for lines in readfile:
                if lines.startswith(">"):
                    ensembl_id = lines.split(" ")[0]
                    seq_fasta.append(f"\n{ensembl_id}\n")
                else:
                    lines = re.sub(r'[^ATGC"]', "", lines)
                    seq_fasta.append(lines.rstrip("\n"))

            seq_fasta[0] = seq_fasta[0].lstrip("\n")
            joined_list = "".join(seq_fasta)
            seq_fasta = joined_list.split("\n")
        return seq_fasta

    # Removing sequences less than 50 nucleotides in length and preparation of binary data: 'A', 'T' are replaced by '0' and 'G', 'C' are replaced by '1'
    def create_binary(self, seq_data):
        seq_binary_list = []
        for index in range(0, len(seq_data)):
            if index % 2 != 0 and len(seq_data[index]) > 50:
                seq_binary_list.append(f"\n{seq_data[index - 1]}\n")
                seq_binary_list.append(
                    seq_data[index].translate(
                        str.maketrans(
                            {
                                "A": str(0),
                                "T": str(0),
                                "G": str(1),
                                "C": str(1),
                                "a": str(0),
                                "t": str(0),
                                "g": str(1),
                                "c": str(1),
                            }
                        )
                    )
                )
        return seq_binary_list

    # Generation of possible k-mer patterns and k should be an integer
    def generate_patterns(self):
        pattern_list = []
        for items in list(product("01", repeat=self.k_mer_value)):
            pattern_list.append("".join(items))
        return pattern_list

    # Finding frequency of generated binary patterns from  binary file
    def get_count(self, seq, pat):
        count = 0
        start = 0
        pos_list = []
        while start < len(seq):
            pos = seq.find(pat, start)
            if pos != -1:
                pos_list.append(pos)
                start = pos + 1
                count = count + 1
            else:
                break
        return count, pos_list

    def find_pattern(self, sequence):
        for pat in tqdm(self.patterns):
            total_freq = 0
            for ensembl_ID, seq in sequence:
                freq, positions = self.get_count(seq, pat)
                total_freq = total_freq + freq
            yield " ".join([pat, str(total_freq), "\n"])


if __name__ == "__main__":

    while True:

        directory = input("Please give the path of your working directory : ")
        isdir = os.path.isdir(directory)
        if isdir == True:
            break
        else:
            print("No such directory available")
            continue

    while True:
        try:
            k_mer = int(input("Process input file for which k-Mer?"))
            break
        except ValueError:
            print("Please provide integer values only")
            continue

    pf_obj = patternFinder(directory, k_mer)
