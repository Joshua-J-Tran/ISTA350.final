from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr

from driver_setup import *

#Categorize country by population. Thompson
def pop_categorizer(df):
    b_1m = []
    b_100m = []
    l_100m = []
    for x in df.index:
        a = int(df.loc[x].iloc[1:-1].astype(float).mean())
        if a < 10000000:
            b_1m.append(df.loc[x].iloc[0])
        elif a < 100000000:
            b_100m.append(df.loc[x].iloc[0])
        else:
            l_100m.append(df.loc[x].iloc[0])
    return [b_1m, b_100m, l_100m]

#read and process the scraped CSv files. Huy Tran
def read_and_process(file_name):
    df = pd.read_csv(file_name, header=0)

    year_columns = [str(year) for year in range(2013, 2023)]
    df.columns = ['Country'] + year_columns

    df.replace('..', np.nan, inplace=True)
    df[year_columns] = df[year_columns].apply(pd.to_numeric)

    return df

#scrape all the given links to csv files. Thompson
def main1():
    driver = create_driver()
    cat_to_url = {'electricity': 'https://databank.worldbank.org/reports.aspx?source=2&series=EG.ELC.ACCS.ZS&country=', \
                        'water': 'https://databank.worldbank.org/reports.aspx?source=2&series=SH.H2O.SMDW.ZS&country=', \
                        'internet': 'https://databank.worldbank.org/reports.aspx?source=2&series=IT.NET.USER.ZS&country=',\
                        'population': 'https://databank.worldbank.org/reports.aspx?source=2&series=SP.POP.TOTL&country=',\
                        "health_expenditure" : 'https://databank.worldbank.org/reports.aspx?source=2&series=SH.XPD.CHEX.GD.ZS&country=',\
                        'gdp':    'https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.MKTP.CD&country=',\
                        'life_expectancy': 'https://databank.worldbank.org/reports.aspx?source=2&series=SP.DYN.LE00.IN&country='}
    df_holder = {}
    for item in cat_to_url:
        df = scraping_world_bank_data(driver, cat_to_url[item])
        df_holder[item] = df
        if df is not None:
            df.to_csv('world_bank_data_' + item + '.csv', index = False)

    driver.quit()

#Sorting the countries by their given information. Huy Tran
def sort(filename):
    df =  read_and_process(filename)
    columns_for_average = df.columns[1:10]  
    df['Average'] = df[columns_for_average].mean(axis=1)
    df_sorted = df.sort_values(by='Average', ascending=False)
    group_size = len(df_sorted) // 3
    group1 = df_sorted.iloc[:group_size]
    group2 = df_sorted.iloc[group_size:2*group_size]
    group3 = df_sorted.iloc[2*group_size:]
    group1_countries = group1.iloc[:, 0].tolist() 
    group2_countries = group2.iloc[:, 0].tolist()
    group3_countries = group3.iloc[:, 0].tolist()

    return group1_countries, group2_countries, group3_countries


#Drawing scatter chart for Huy Tran
def scatter1(top_countries, mid_countries, bottom_countries):
    health_expenditure_df = read_and_process('world_bank_data_health_expenditure.csv')
    gdp_df = read_and_process('world_bank_data_gdp.csv')

    year_columns = [str(year) for year in range(2013, 2023)]
    health_expenditure_df['Avg_Health_Expenditure'] = health_expenditure_df[year_columns].mean(axis=1)
    gdp_df['Avg_GDP_Trillions'] = gdp_df[year_columns].mean(axis=1) / 1e12
    merged_df = pd.merge(health_expenditure_df[['Country', 'Avg_Health_Expenditure']], gdp_df[['Country', 'Avg_GDP_Trillions']], on='Country')
    plt.figure(figsize=(8, 12))
    for country in merged_df['Country']:
        if country in top_countries:
            plt.scatter(merged_df[merged_df['Country'] == country]['Avg_Health_Expenditure'], merged_df[merged_df['Country'] == country]['Avg_GDP_Trillions'], color='red', alpha=0.5, label='Top GDP Countries' if country == top_countries[0] else "")
        elif country in mid_countries:
            plt.scatter(merged_df[merged_df['Country'] == country]['Avg_Health_Expenditure'], merged_df[merged_df['Country'] == country]['Avg_GDP_Trillions'], color='yellow', alpha=0.5, label='Mid GDP Countries' if country == mid_countries[0] else "")
        elif country in bottom_countries:
            plt.scatter(merged_df[merged_df['Country'] == country]['Avg_Health_Expenditure'], merged_df[merged_df['Country'] == country]['Avg_GDP_Trillions'], color='green', alpha=0.5, label='Bottom GDP Countries' if country == bottom_countries[0] else "")

    plt.title('Average Health Expenditure (% of GDP) vs Average GDP (2013-2022)')
    plt.xlabel('Average Health Expenditure (% of GDP)')
    plt.ylabel('Average GDP (Trillions of USD)')
    plt.ylim(0, 20)
    plt.grid(True)
    plt.legend()
    plt.show()

#Bar chart. Huy Tran
def bar1(group1_countries, group2_countries, group3_countries):
    df = read_and_process('world_bank_data_electricity.csv')
    year_columns = df.columns[1:10]
    df['Avg_Electricity_Access'] = df[year_columns].mean(axis=1)
    electricity_df = df[['Country', 'Avg_Electricity_Access']]
    electricity_df.set_index('Country', inplace=True)
    group1_df = electricity_df.reindex(group1_countries)
    group2_df = electricity_df.reindex(group2_countries)
    group3_df = electricity_df.reindex(group3_countries)

    final_df = pd.concat([group1_df, group2_df, group3_df])

    plt.figure(figsize=(15, 10))
    plt.bar(range(len(final_df)), final_df['Avg_Electricity_Access'], color=['blue']*len(group1_df) + ['green']*len(group2_df) + ['red']*len(group3_df))

    plt.ylabel('Average Electricity Access (%)')
    plt.xlabel('Countries')
    plt.title('Average Access to Electricity by Country Group')
    plt.show()

def chart(group1_countries, group2_countries, group3_countries):
    life_expectancy_df = read_and_process('world_bank_data_life_expectancy.csv')
    health_expenditure_df = read_and_process('world_bank_data_health_expenditure.csv')
    year_columns = life_expectancy_df.columns[1:]

    group1_life = life_expectancy_df[life_expectancy_df['Country'].isin(group1_countries)]
    group2_life = life_expectancy_df[life_expectancy_df['Country'].isin(group2_countries)]
    group3_life = life_expectancy_df[life_expectancy_df['Country'].isin(group3_countries)]

    group1_health = health_expenditure_df[health_expenditure_df['Country'].isin(group1_countries)]
    group2_health = health_expenditure_df[health_expenditure_df['Country'].isin(group2_countries)]
    group3_health = health_expenditure_df[health_expenditure_df['Country'].isin(group3_countries)]

    avg_group1_life = group1_life[year_columns].mean()
    avg_group2_life = group2_life[year_columns].mean()
    avg_group3_life = group3_life[year_columns].mean()

    avg_group1_health = group1_health[year_columns].mean()
    avg_group2_health = group2_health[year_columns].mean()
    avg_group3_health = group3_health[year_columns].mean()

    fig, ax1 = plt.subplots(figsize=(15, 10))

    ax1.plot(year_columns, avg_group1_life, color='blue', label='Group 1 Life Expectancy')
    ax1.plot(year_columns, avg_group2_life, color='green', label='Group 2 Life Expectancy')
    ax1.plot(year_columns, avg_group3_life, color='red', label='Group 3 Life Expectancy')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Life Expectancy')

    ax2 = ax1.twinx()
    ax2.plot(year_columns, avg_group1_health, 'b--', label='Group 1 Health Expenditure')
    ax2.plot(year_columns, avg_group2_health, 'g--', label='Group 2 Health Expenditure')
    ax2.plot(year_columns, avg_group3_health, 'r--', label='Group 3 Health Expenditure')
    ax2.set_ylabel('Health Expenditure (% of GDP)')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Average Life Expectancy and Health Expenditure Changes Over the Years by Group')
    plt.show()

def chart2(group1_countries, group2_countries, group3_countries):
    life_expectancy_df = read_and_process('world_bank_data_life_expectancy.csv')
    internet_access_df = read_and_process('world_bank_data_internet.csv')
    year_columns = life_expectancy_df.columns[1:]

    group1_life = life_expectancy_df[life_expectancy_df['Country'].isin(group1_countries)]
    group2_life = life_expectancy_df[life_expectancy_df['Country'].isin(group2_countries)]
    group3_life = life_expectancy_df[life_expectancy_df['Country'].isin(group3_countries)]

    group1_internet = internet_access_df[internet_access_df['Country'].isin(group1_countries)]
    group2_internet = internet_access_df[internet_access_df['Country'].isin(group2_countries)]
    group3_internet = internet_access_df[internet_access_df['Country'].isin(group3_countries)]

    avg_group1_life = group1_life[year_columns].mean()
    avg_group2_life = group2_life[year_columns].mean()
    avg_group3_life = group3_life[year_columns].mean()

    avg_group1_internet = group1_internet[year_columns].mean()
    avg_group2_internet = group2_internet[year_columns].mean()
    avg_group3_internet = group3_internet[year_columns].mean()

    fig, ax1 = plt.subplots(figsize=(15, 10))

    ax1.plot(year_columns, avg_group1_life, color='blue', label='Group 1 Life Expectancy')
    ax1.plot(year_columns, avg_group2_life, color='green', label='Group 2 Life Expectancy')
    ax1.plot(year_columns, avg_group3_life, color='red', label='Group 3 Life Expectancy')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Life Expectancy')

    ax2 = ax1.twinx()
    ax2.plot(year_columns, avg_group1_internet, 'b--', label='Group 1 Internet Access')
    ax2.plot(year_columns, avg_group2_internet, 'g--', label='Group 2 Internet Access')
    ax2.plot(year_columns, avg_group3_internet, 'r--', label='Group 3 Internet Access')
    ax2.set_ylabel('Internet Access (% of Population)')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Average Life Expectancy and Internet Access Changes Over the Years by Group')
    plt.show()

def main2():
    top,mid,bottom = sort ('world_bank_data_population.csv')
    top2,mid2,bottom2 = sort ('world_bank_data_gdp.csv')
    top3,mid3,bottom3 = sort ('world_bank_data_health_expenditure.csv')
    group1, group2, group3 = sort('world_bank_data_internet.csv')
    group4, group5, group6 = sort('world_bank_data_life_expectancy.csv')
    print (top2[:3])
    print (top3[:3])
    scatter1(top2,mid2,bottom2)
    chart(top2,mid2,bottom2)
    chart2(top2,mid2,bottom2)
    scatter2()

#Script to draw the scatter charts and regression line. Thompson
def scatter2():
    dict2 = {}
    y_label_dict = {'electricity': 'Coverage percentage (%)', 'water': 'Coverage percentage (%)', 'internet': 'Coverage percentage (%)', 'population': 'Population (billion)', 'gdp': 'Current GDP (trillion USD)', 'life_expectancy': 'Age(year)'}
    pop_cat = pop_categorizer(read_and_process('world_bank_data_population.csv'))
    print(pop_cat)
    for i in ['electricity', 'water', 'internet', 'population', 'gdp', 'life_expectancy']:
        dict2[i] = read_and_process('world_bank_data_' + i + '.csv')
        scatter2_foreach(pop_cat,dict2[i], 'World Average ' + i.capitalize() + ' progress', y_label_dict[i])
    
#individual scatter. Thompson
def scatter2_foreach(pop_cat, df, title, y_lb):
    marker = ['o', 'x', 'd', 'p']
    color = ['red', 'black', 'blue', 'green']
    
    df.index = df.iloc[:, 0]
    for j in range(4):
        worlds = []
        if j == 0:
            for i in range(2013,2022):
                worlds.append(df.loc[:,str(i)].dropna().mean())
        else:
            for i in range(2013,2022):
                print('temp', df.index)
                worlds.append(df.loc[list(pop_cat[j-1]),str(i)].dropna().mean())            
        print(worlds)
        sr = pd.Series(index = [i for i in range(2013, 2022)], data = worlds)
        sr.plot(linestyle = '', marker = marker[j], markersize = 6, color = color[j])
        X = sm.add_constant(sr.index)
        model = sm.OLS(sr, X)
        result = model.fit()
        y = result.params['x1'] * sr.index + result.params['const']
        plt.plot(sr.index, y,color = color[j], linewidth=4)
    plt.title(title)
    plt.ylabel(y_lb)
    plt.xlabel('Year')
    plt.legend(['world', 'world reg', 'below 10 million', 'b1m reg', 'below 100 million','b100m reg', 'above 100 million', 'a100m reg'])
    plt.show()
    return result


    
if __name__ == "__main__":
    main1()
    main2()
    

