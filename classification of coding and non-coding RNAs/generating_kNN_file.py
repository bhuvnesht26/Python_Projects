#!/usr/bin/env python3
# coding: utf-8
# Date: Mon, Jun 7, 2021

from itertools import product
import pandas as pd
import re
import csv
import time
import os


class generate_kNN_file:
    def __init__(self, directory):
        self.working_dir = directory
        self.patterns = self.generate_patterns(5)
        t1 = time.time()
        for inputfile in os.listdir(self.working_dir):

            if inputfile.endswith(".fa") or inputfile.endswith(".fasta"):
                genome_type = inputfile.split("_")[-1].split(".")[0]
                file_name = inputfile.split(".")[0]
                seq_list = self.read_file(os.path.join(self.working_dir, inputfile))
                binary_list = self.create_binary(seq_list)

                if genome_type == "cds":
                    outputfile = os.path.join(self.working_dir, f"{file_name}.csv")
                    # print(outputfile)
                    self.write_csv_file(outputfile, binary_list, 1)
                    df_1 = pd.read_csv(outputfile, header=None)
                    print(df_1)
                else:
                    outputfile = os.path.join(self.working_dir, f"{file_name}.csv")
                    # print(outputfile)
                    self.write_csv_file(outputfile, binary_list, 2)
                    df_2 = pd.read_csv(outputfile, header=None)
        df_concat = pd.concat([df_1, df_2], axis=0).reset_index(drop=True)
        df_concat.to_csv("inputfile_for_ML_fivemer.csv", header=None)

        t2 = time.time()
        print(f"It took {t2 - t1} seconds to process.")

    def read_file(self, fasta_file):
        data_list = []
        sequence_list = []
        with open(fasta_file, "r") as readfile:
            for lines in readfile:
                if lines.startswith(">"):
                    joined_sequence = "".join(sequence_list)
                    if joined_sequence != "":
                        data_list.append(joined_sequence)
                        sequence_list.clear()
                    ensembl_id = lines.split(" ")[0]
                    data_list.append("\n" + ensembl_id + "\n")
                else:
                    sequence = lines.rstrip("\n")

                    sequence = re.sub(r'[^ATGC"]', "", sequence)
                    sequence_list.append(sequence)
            else:
                data_list.append("".join(sequence_list))
                sequence_list.clear()
        return data_list

    def create_binary(self, seq_data):
        output = []
        for index in range(0, len(seq_data)):
            if index % 2 != 0 and len(seq_data[index]) > 50:
                output.append(seq_data[index - 1])
                output.append(
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

        output_list = list(
            zip(
                [output[x] for x in range(0, len(output)) if x % 2 == 0],
                [output[x] for x in range(0, len(output)) if x % 2 != 0],
            )
        )

        return output_list

    def generate_patterns(self, k):
        pattern_list = []
        for items in list(product("01", repeat=k)):
            pattern_list.append("".join(items))
        return pattern_list

    def get_count(self, seq, pat):
        count = 0
        start = 0
        while start < len(seq):
            pos = seq.find(pat, start)
            if pos != -1:
                start = pos + 1
                count = count + 1
            else:
                break
        return count

    def write_csv_file(self, file, sequence, number):
        with open(file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for ensembl_ID, seq in sequence:

                freq_list = []
                for pat in self.patterns:
                    freq = self.get_count(seq, pat)
                    freq_list.append(freq)
                freq_list.append(number)
                writer.writerow(freq_list)


if __name__ == "__main__":

    directory = input(
        "Please give the path of directory containing CDS and ncRNA files: "
    )
    file_obj = generate_kNN_file(directory)
