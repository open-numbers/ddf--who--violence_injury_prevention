# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os.path as osp
import json
from ddf_utils.str import to_concept_id


def is_country(df):
    a = df['Admin1'].fillna(0).map(str)
    b = df['SubDiv'].fillna('')
    c = a + b
    res = c.map(lambda x: True if x == '0.0' else False)
    return res


def preprocess(df, subset):
    cols = ['Country', 'Year', 'Cause', 'Sex', 'Deaths1', 'Deaths2', 'Deaths3', 'Deaths4', 'Deaths5',
            'Deaths6', 'Deaths7', 'Deaths8', 'Deaths9', 'Deaths10', 'Deaths11',
            'Deaths12', 'Deaths13', 'Deaths14', 'Deaths15', 'Deaths16', 'Deaths17',
            'Deaths18', 'Deaths19', 'Deaths20', 'Deaths21', 'Deaths22', 'Deaths23',
            'Deaths24', 'Deaths25', 'Deaths26', 'IM_Deaths1', 'IM_Deaths2',
            'IM_Deaths3', 'IM_Deaths4']

    df1 = df.copy()
    df1['is_country'] = is_country(df1)
    df1 = df1[df1.is_country == True][cols]

    df2 = df1[df1.Cause.isin(subset)].copy()

    return df2


def extract(df, concepts_mapping, cause_mapping, check_completeness=[], threshold=0.9):
    df.columns = df.columns.map(concepts_mapping)
    df.cause = df.cause.map(cause_mapping)

    cause_mapping_grouped = dict()
    for k, v in cause_mapping.items():
        if v not in cause_mapping_grouped.keys():
            cause_mapping_grouped[v] = [k]
        else:
            cause_mapping_grouped[v].append(k)

    res = []

    for cause, idx in df.groupby('cause').groups.items():
        df_cause = df.loc[idx].copy()
        df_cause = df_cause.drop('cause', axis=1)
        if cause in check_completeness:
            min_count = round(threshold * len(cause_mapping_grouped[cause]))
        else:
            min_count =  -1
        df_new = df_cause.groupby(['country', 'year', 'sex']).sum(min_count=min_count)
        df_new['cause'] = cause
        df_new = df_new.set_index('cause', append=True)
        df_new.reorder_levels(['country', 'year', 'cause', 'sex'])
        res.append(df_new)

    return pd.concat(res)


def rate(deaths, pop, pop_div, gs):
    params = []
    for i in gs:
        if isinstance(i['spec'], str):
            pop_g = pop['population_age_{}_years'.format(i['spec'].replace('-', '_'))]
            deaths_g = deaths['deaths_age_{}_years'.format(i['spec'].replace('-', '_'))]
        else:
            pop_g = pop[i['spec']].sum(axis=1)
            deaths_g = (deaths[[x.replace('population', 'deaths').replace('over', 'above')
                                for x in i['spec']]]
                        .sum(axis=1))

        a, b = pop_g.align(deaths_g)
        std_pop_g = pop_div.loc[0, i['std']]
        params.append([a, b, std_pop_g])

    total_std_pop = sum(x[-1] for x in params)

    res = []
    for ps in params:
        res.append(calc(ps[1], ps[0], ps[2], total_std_pop))

    r0 = res.pop()
    for r in res:
        r0 = r0.append(r)
    r0 = r0.groupby(level=[0, 1, 2]).sum(min_count=len(gs)) * 100000  # all group should have data.
    return r0.replace(0.0, np.nan).dropna()


def calc(death, pop, stdPop, stdPopSum):
    return (death / pop) * (stdPop / stdPopSum)


def rate_allage(deaths, pop, pop_div):
    d = deaths['deaths_all_ages']
    p = pop['population_all_ages']

    return d / p * 100000


def rate_014(deaths, pop, pop_div):
    gs = [
        {'spec': '0', 'std': '0'},
        {
            'spec': [
                'population_age_1_year',
                'population_age_2_years', 'population_age_3_years',
                'population_age_4_years'
            ],
            'std': '01-04'},
        {'spec': '5_9', 'std': '5-9'},
        {'spec': '10_14', 'std': '10-14'}
    ]

    return rate(deaths, pop, pop_div, gs)


def rate_1529(deaths, pop, pop_div):
    gs = [
        {'spec': '15_19', 'std': '15-19'},
        {'spec': '20_24', 'std': '20-24'},
        {'spec': '25_29', 'std': '25-29'}
    ]

    return rate(deaths, pop, pop_div, gs)


def rate_3044(deaths, pop, pop_div):
    gs = [
        {'spec': '30_34', 'std': '30-34'},
        {'spec': '35_39', 'std': '35-39'},
        {'spec': '40_44', 'std': '40-44'}
    ]

    return rate(deaths, pop, pop_div, gs)


def rate_4559(deaths, pop, pop_div):
    gs = [
        {'spec': '45_49', 'std': '45-49'},
        {'spec': '50_54', 'std': '50-54'},
        {'spec': '55_59', 'std': '55-59'}
    ]

    return rate(deaths, pop, pop_div, gs)


def rate_60plus(deaths, pop, pop_div):
    gs = [
        {'spec': '60_64', 'std': '60-64'},
        {'spec': '65_69', 'std': '65-69'},
        {'spec': '70_74', 'std': '70-74'},
        {'spec': '75_79', 'std': '75-79'},
        {'spec': '80_84', 'std': '80-84'},
        {'spec': [
            'population_age_85_89_years',
            'population_age_90_94_years',
            'population_age_95_years_and_over'
        ], 'std': '85+'}
    ]

    return rate(deaths, pop, pop_div, gs)


def read_mort_source():
    icd7_mapping = json.load(open('./icd7.json'))
    icd8_mapping = json.load(open('./icd8.json'))
    icd9_mapping = json.load(open('./icd9.json'))
    icd10_mapping = json.load(open('./icd10.json'))
    concept_mapping = json.load(open('./concept_mapping.json'))

    icd7 = pd.read_csv('../source/morticd07.zip')
    df1 = preprocess(icd7, icd7_mapping.keys())
    df1 = extract(df1, concept_mapping, icd7_mapping)

    icd8 = pd.read_csv("../source/morticd08.zip")
    df2 = preprocess(icd8, icd8_mapping.keys())
    df2 = extract(df2, concept_mapping, icd8_mapping)

    icd9 = pd.read_csv('../source/morticd9.zip')
    df3 = preprocess(icd9, icd9_mapping.keys())
    df3 = extract(df3, concept_mapping, icd9_mapping)

    icd101 = pd.read_csv('../source/Morticd10_part1.zip')
    df4 = preprocess(icd101, icd10_mapping.keys())
    df4 = extract(df4, concept_mapping, icd10_mapping)

    icd102 = pd.read_csv('../source/Morticd10_part2.zip')
    df5 = preprocess(icd102, icd10_mapping.keys())
    df5 = extract(df5, concept_mapping, icd10_mapping)

    df_all = pd.concat([df1, df2, df3, df4, df5])
    return df_all


def read_population_source():
    pop_div = pd.read_csv("../source/pop.csv")
    pop_concepts = pd.read_csv('../source/concepts_pop.csv')
    pop_concepts = pop_concepts.dropna(how='all', axis=1)
    pop_concepts['concept'] = pop_concepts['Column name'].map(to_concept_id)
    pop_concepts.loc[3:, 'concept'] = pop_concepts.loc[3:, 'Content'].map(lambda x: to_concept_id(x.replace("at ", "")))
    pop_concept_mapping = pop_concepts.set_index('Column name')['concept'].to_dict()
    pop = pd.read_csv('../source/Pop.zip')

    pop["is_country"] = is_country(pop)
    pop = pop[pop.is_country == True]

    cols = ['Country', 'Year', 'Sex', 'Pop1', 'Pop2',
            'Pop3', 'Pop4', 'Pop5', 'Pop6', 'Pop7', 'Pop8', 'Pop9', 'Pop10',
            'Pop11', 'Pop12', 'Pop13', 'Pop14', 'Pop15', 'Pop16', 'Pop17', 'Pop18',
            'Pop19', 'Pop20', 'Pop21', 'Pop22', 'Pop23', 'Pop24', 'Pop25', 'Pop26',
            'Lb']
    pop = pop[cols]
    pop.columns = pop.columns.map(pop_concept_mapping)
    pop_ = pop.copy()
    pop_div_t = pop_div.set_index('All ages').T.reset_index(drop=True)

    pop_ = pop_.rename(columns={'population_age1_year': 'population_age_1_year'})
    pop_ = pop_.set_index(['country', 'year', 'sex'])
    return pop_
