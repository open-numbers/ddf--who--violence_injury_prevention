# -*- coding: utf-8 -*-

import pandas as pd
from ddf_utils.str import to_concept_id


def main():
    data = pd.read_excel('../source/injury_mortality_trend_tables.xls', sheet_name="Rates", skiprows=4)

    # manually set the cause entity domain name->concept mapping
    m = {'Road traffic accidents': 'traffic', 'Homicide': 'homicide', 'Self-inflicted injuries': 'suicide'}

    data = data.rename(columns={'cause of death': 'cause'})
    data['cause'] = data['cause'].map(m)

    # datapoints
    dps = data.copy()
    dps = dps.drop(['ICD', 'name'], axis=1)
    dps = dps.set_index(['country', 'year', 'cause', 'sex'])
    cols = dps.columns
    for c in dps:
        c_ = to_concept_id(c)
        df = dps[c].copy()
        df.name = c_
        df.reset_index().dropna().to_csv(
            f"../../ddf--datapoints--{c_}--by--country--year--cause--sex.csv", index=False)

    # country
    country = data[['country', 'name']].drop_duplicates(subset=['country'], keep='first')
    country.to_csv('../../ddf--entities--country.csv', index=False)

    # sex
    sex = pd.DataFrame([[0, 'Both sexes'],
                        [1, 'male'],
                        [2, 'female']], columns=['sex', 'name'])
    sex.to_csv('../../ddf--entities--sex.csv', index=False)

    # cause
    cause = pd.DataFrame.from_dict(m, orient='index').reset_index()
    cause.columns = ['name', 'cause']
    cause.to_csv('../../ddf--entities--cause.csv', index=False)

    # concepts
    cont = cols.map(to_concept_id)
    cont_df = pd.DataFrame.from_dict({'concept': cont, 'name': cols})
    cont_df['concept_type'] = 'measure'

    ent_df = pd.DataFrame([['country', 'Country'],
                           ['sex', 'Sex'],
                           ['cause', 'Cause']], columns=['concept', 'name'])
    ent_df['concept_type'] = 'entity_domain'

    other_df = pd.DataFrame([['year', 'Year', 'time'],
                             ['name', 'Name', 'string']], columns=['concept', 'name', 'concept_type'])

    concepts_df = pd.concat([cont_df, ent_df, other_df], sort=False)
    concepts_df.to_csv('../../ddf--concepts.csv', index=False)


if __name__ == '__main__':
    main()
    print('Done.')
