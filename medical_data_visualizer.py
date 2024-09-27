import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv('medical_examination.csv')

# 2. Create the overweight column in the df variable
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3. Normalize data by making 0 always good and 1 always bad for cholesterol and gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)

# 4. Draw the Categorical Plot in the draw_cat_plot function
def draw_cat_plot():
    # 5. Create a DataFrame for the cat plot using pd.melt with values from cholesterol, gluc, smoke, alco, active, and overweight in the df_cat variable.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Convert the data into long format and create a chart that shows the value counts of the categorical features using sns.catplot()
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar")

    # 8. Get the figure for the output and store it in the fig variable
    fig.fig

    # 9. Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig.fig

# 10. Draw the Heat Map in the draw_heat_map function
def draw_heat_map():
    # 11. Clean the data by filtering out patient segments with incorrect data (2.5th and 97.5th percentiles for height and weight)
    df_heat = df[
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr()
    
    # Convert all values to float, ensuring that they are treated as numerical types
    corr = corr.astype(float)
    corr = corr.where(corr != -0.0, 0.0)  # Remove -0.0

    # 13. Generate a mask for the upper triangle and store it in the mask variable
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15. Plot the correlation matrix using sns.heatmap()
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", ax=ax, cmap="coolwarm", vmax=.3, center=0, square=True, linewidths=.5)

    # 16. Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
