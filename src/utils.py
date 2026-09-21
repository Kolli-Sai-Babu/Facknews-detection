import os
import zipfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

def download_and_extract_datasets(target_dir="dataset"):
    """
    Downloads Fake.csv.zip and True.csv.zip from the Raghuls-github repository mirror
    and extracts them to the specified dataset directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    urls = {
        "Fake.csv.zip": "https://github.com/Raghuls-github/Fake-News-Detection/raw/master/Fake.csv.zip",
        "True.csv.zip": "https://github.com/Raghuls-github/Fake-News-Detection/raw/master/True.csv.zip"
    }
    
    # Try downloading and extracting
    for filename, url in urls.items():
        zip_path = os.path.join(target_dir, filename)
        csv_name = filename.replace(".zip", "")
        csv_path = os.path.join(target_dir, csv_name)
        
        # Check if CSV already exists to skip download
        if os.path.exists(csv_path):
            print(f"{csv_name} already exists. Skipping download.")
            continue
            
        print(f"Downloading {filename} from {url}...")
        try:
            # Add user agent headers to prevent HTTP 403 Forbidden errors
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
                
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                
            # Clean up the zip file
            os.remove(zip_path)
            print(f"Successfully processed {csv_name}.")
        except Exception as e:
            print(f"Failed to download or extract {filename} from {url}. Error: {e}")
            # Try main branch fallback
            fallback_url = url.replace("/master/", "/main/")
            print(f"Attempting fallback download from {fallback_url}...")
            try:
                req = urllib.request.Request(
                    fallback_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                    out_file.write(response.read())
                    
                print(f"Extracting {filename} (fallback)...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                os.remove(zip_path)
                print(f"Successfully processed {csv_name} (fallback).")
            except Exception as e_fallback:
                print(f"Fallback also failed: {e_fallback}. Please download Fake.csv and True.csv manually.")
                raise e_fallback

def load_data(dataset_dir="dataset"):
    """
    Loads Fake.csv and True.csv, adds labels (0 for Fake, 1 for True),
    merges them, shuffles the resulting dataframe, and resets the index.
    """
    fake_path = os.path.join(dataset_dir, "Fake.csv")
    true_path = os.path.join(dataset_dir, "True.csv")
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("CSV files not found. Attempting to download them...")
        download_and_extract_datasets(dataset_dir)
        
    print("Loading datasets...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add label column
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Merge datasets
    merged_df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Shuffle dataset
    shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset loaded. Total records: {len(shuffled_df)} (Fake: {len(fake_df)}, True: {len(true_df)})")
    return shuffled_df

def generate_eda_visualizations(df, output_dir="notebook/plots"):
    """
    Performs EDA and saves visual plots:
    1. Label distribution
    2. Document length distribution
    3. Word clouds for real and fake news
    4. Most common words bar chart
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Label Distribution Plot
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='label', data=df, palette={0: "#e74c3c", 1: "#2ecc71", "0": "#e74c3c", "1": "#2ecc71"})
    ax.set_xticklabels(['Fake (0)', 'True (1)'])
    plt.title('Distribution of Fake vs Real News')
    plt.xlabel('Label')
    plt.ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() + 100),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_distribution.png"), dpi=150)
    plt.close()
    
    # 2. News Length Distribution
    # Add length columns
    df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[df['label'] == 0]['text_len'], label='Fake News', shade=True, color="#e74c3c")
    sns.kdeplot(df[df['label'] == 1]['text_len'], label='True News', shade=True, color="#2ecc71")
    plt.xlim(0, 1500) # Limit x-axis to show the bulk of the distribution
    plt.title('News Length Distribution (Word Count)')
    plt.xlabel('Number of Words')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "length_distribution.png"), dpi=150)
    plt.close()
    
    # 3. Word Clouds
    # We generate word clouds from text directly to avoid doing it inline in notebooks or report generators
    # Wordcloud for Fake News
    print("Generating Word Clouds...")
    fake_text = " ".join(df[df['label'] == 0]['title'].astype(str).tolist()[:1000])
    wordcloud_fake = WordCloud(width=800, height=400, background_color='white', 
                               colormap='Reds', max_words=100).generate(fake_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_fake, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Fake News Headlines', fontsize=16)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, "wordcloud_fake.png"), dpi=150)
    plt.close()
    
    # Wordcloud for True News
    true_text = " ".join(df[df['label'] == 1]['title'].astype(str).tolist()[:1000])
    wordcloud_true = WordCloud(width=800, height=400, background_color='white', 
                               colormap='Greens', max_words=100).generate(true_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_true, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - True News Headlines', fontsize=16)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, "wordcloud_true.png"), dpi=150)
    plt.close()
    
    # 4. Most Common Words in Titles
    print("Calculating most common words...")
    all_fake_words = " ".join(df[df['label'] == 0]['title'].astype(str)).lower().split()
    all_true_words = " ".join(df[df['label'] == 1]['title'].astype(str)).lower().split()
    
    # Filter stopwords basic check
    stopwords_list = {'to', 'the', 'of', 'in', 'and', 'a', 'on', 'for', 'with', 'is', 'as', 'at', 'by', 'that', 'from', 'this', 'about', 'be', 'it'}
    fake_word_freq = Counter([w for w in all_fake_words if w.isalpha() and w not in stopwords_list]).most_common(15)
    true_word_freq = Counter([w for w in all_true_words if w.isalpha() and w not in stopwords_list]).most_common(15)
    
    # Plot most common words
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fake
    fake_words, fake_counts = zip(*fake_word_freq)
    sns.barplot(x=list(fake_counts), y=list(fake_words), ax=axes[0], palette="Reds_r")
    axes[0].set_title('Top 15 Common Words in Fake News Titles')
    axes[0].set_xlabel('Count')
    
    # True
    true_words, true_counts = zip(*true_word_freq)
    sns.barplot(x=list(true_counts), y=list(true_words), ax=axes[1], palette="Greens_r")
    axes[1].set_title('Top 15 Common Words in True News Titles')
    axes[1].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_words_comparison.png"), dpi=150)
    plt.close()
    print("EDA Visualizations saved to:", output_dir)
