import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_wordcloud(input_file):
    df = pd.read_csv(input_file)
    text = " ".join(df["CleanedTitle"])

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    generate_wordcloud("../Data/cleaned_data.csv")
