'''
Script used for processing csv file from water
'''
import os
import sys
import pandas as pd
import numpy as np
import argparse

from scipy.stats import multivariate_normal

parser = argparse.ArgumentParser(description='Pre-processing Water Mass Spectrometry data',
                                 epilog='This is a command line program designed for pre-processing data from Water'
                                        'Mass spectrometry instrument with a desired format that can be readily used'
                                        'for Graph neural network training.')


parser.add_argument('path', type=str,
                    help='Path to the csv file containing the data from water mass spectrometry instrument')

parser.add_argument('-o', '--output', type=str, required=False,
                    help='Path to the output file, if not provided, the output file will be saved in the same directory'
                    'as the input file with the name "protein_probability.csv')

parser.add_argument('-f', '--filter-file', type=str, required=False,
                    help='the file used for filtering the what proteins to keep,'
                         'expecting a tsv file with first column as protein accession and second column as protein name')


args = parser.parse_args()

df = pd.read_csv(args.path, usecols=['protein.Accession', 'protein.dataBaseType', 'protein.score'])
df['protein.score'] = np.log(df['protein.score'] + 1)


# filter protein if needed
if args.filter_file:
    protein_list = pd.read_table(args.filter_file, sep='\t')
    first_col = protein_list.columns[0]
    flag = np.logical_or(df['protein.Accession'].isin(protein_list[first_col]), df['protein.dataBaseType'] == 'Random') # both Swissprot and decoy
    df = df[flag]

d0 = df[df['protein.dataBaseType'] == 'Random']['protein.score'].values
d = df[df['protein.dataBaseType'] == 'Regular']['protein.score'].values
d = np.sort(d)


def Estep(lis1):
    m0 = lis1[0]
    m1 = lis1[1]
    cov0 = lis1[2]
    cov1 = lis1[3]
    pi = lis1[4]

    pt0 = multivariate_normal.pdf(d, mean=m0, cov=cov0)
    pt1 = multivariate_normal.pdf(d, mean=m1, cov=cov1)
    w0 = (1 - pi) * pt0
    w1 = pi * pt1
    eval1 = w1 / (w1 + w0)
    return eval1

def Mstep(eval1):
    mu0 = (np.dot((1 - eval1), d) + np.sum(d0)) / (np.sum(1 - eval1) + len(d0))
    mu1 = (np.dot(eval1, d)) / (np.sum(eval1))
    # the weighted variance was calculated from the target protein set and the decoy set
    # variance from the target protein set
    cov0_part1 = np.multiply((d - mu0).T, d - mu0)
    # variance from the decoy set
    cov0_part2 = np.multiply((d0 - mu0).T, d0 - mu0)
    # weighted variance
    s0_num = np.dot(cov0_part1, 1 - eval1) + np.sum(cov0_part2)
    s0_denom = np.sum(1 - eval1) + len(d0)
    s0 = s0_num / s0_denom

    # for variance of the target protein set
    # only calculate the weighted variance from the target protein set
    cov1_part1 = np.multiply((d - mu1).T, d - mu1)
    s1_num = np.dot(cov1_part1, eval1)
    s1_denom = np.sum(eval1)
    s1 = s1_num / s1_denom

    pi = sum(eval1) / len(d)
    lis2 = [mu0, mu1, s0, s1, pi]
    return lis2


# initialize the parameters
m0 = np.mean(d)
m1 = np.mean(d)
cov0 = np.cov(np.transpose(d0))
cov1 = np.cov(np.transpose(d))
pi = 0.5
max_iter = 1000
lis1 = [m0, m1, cov0, cov1, pi]

# EM algorithm
for i in range(0, max_iter):
    lis2 = Mstep(Estep(lis1))
    if np.sum(np.abs(np.array(lis2) - np.array(lis1))) < 1e-8:
        break
    lis1 = lis2


def calculate_post_prot_prob(x, mu0, mu1, s0, s1, pi):
    Z0 = multivariate_normal(mu0, s0)
    Z1 = multivariate_normal(mu1, s1)
    return pi * Z1.pdf(x)/((1-pi) * Z0.pdf(x) + pi * Z1.pdf(x))


df['protein_probability'] = calculate_post_prot_prob(df['protein.score'].values,
                                                     mu0=lis1[0],
                                                     mu1=lis1[1],
                                                     s0=lis1[2],
                                                     s1=lis1[3],
                                                     pi=lis1[4])
# filter out the decoy proteins
df_regular = df[df['protein.dataBaseType'] == 'Regular']
# get the max protein probability for each protein
max_values = df_regular.groupby('protein.Accession')['protein_probability'].max()
output = pd.DataFrame({'protein.Accession': max_values.index, 'protein_probability': max_values.values})
# add the proteins that are not in the dat file with probability 0
protein_not_in_dat = protein_list[~protein_list[first_col].isin(output['protein.Accession'])][first_col].values
additional_rows = ({
    'protein.Accession': protein_not_in_dat,
    'protein_probability': np.zeros(len(protein_not_in_dat))
})
output = pd.concat([output, pd.DataFrame(additional_rows)], ignore_index=True)

if args.output:
    output.to_csv(args.output, index=False)
else:
    folder = os.path.dirname(args.path)
    filename = os.path.basename(args.path.split('.')[0]) + '_with_protein_probability.csv'
    output.to_csv(os.path.join(folder, filename), index=False)

print('Processing Done!')