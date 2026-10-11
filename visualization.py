import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations(csv_path):
    print("Adatok betöltése és ábrák generálása...")
    df = pd.read_csv(csv_path)
    
    os.makedirs('plots', exist_ok=True)
    
    # 1. Mérgező eloszlás
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='MIVS', palette='Set2')
    plt.title('Mérgező és Nem mérgező képek aránya')
    plt.xticks([0, 1], ['Nem mérges (0)', 'Mérges (1)'])
    plt.ylabel('Képek száma')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.savefig('plots/venomous_distribution.png')
    plt.close()
    
    # 2. Top 15 faj
    plt.figure(figsize=(10, 6))
    top_species = df['binomial_name'].value_counts().head(15)
    sns.barplot(y=top_species.index, x=top_species.values, palette='mako')
    plt.title('Top 15 leggyakoribb kígyófaj')
    plt.xlabel('Képek száma')
    plt.ylabel('Faj neve')
    plt.tight_layout()
    plt.savefig('plots/top_species_distribution.png')
    plt.close()

    print("Az ábrák sikeresen elkészültek a 'plots' mappában!")

if __name__ == "__main__":
    create_visualizations("merged_snake_metadata.csv")