from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


base =  Path(__file__).parent
movies = pd.read_csv(base / "datasets" / "movies.csv")
ratings = pd.read_csv(base / "datasets" / "ratings.csv")



def diversify(df, h, max_genre_frac=0.3):
    max_per_genre = max(1, int(h * max_genre_frac))
    kept = []
    genre_counts = {}
    for _, row in df.iterrows():
        genres = row["genres"].split("|")
        if any(genre_counts.get(g, 0) >= max_per_genre for g in genres):
            continue
        kept.append(row)
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
        if len(kept) >= h:
            break
    return pd.DataFrame(kept)

def popularity(h = None, threshold = 0.3):
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])

    prior_weight = stats["count"].mean()
    prior_mean = stats["mean"].mean()
    stats["score"] = ((stats["count"] * stats["mean"] + prior_weight * prior_mean) / (stats["count"] + prior_weight))

    stats = stats.reset_index().merge(movies, on="movieId")
    stats = stats.sort_values("score", ascending=False)

    if h is not None:
        topQuan = stats["count"].quantile(1 - threshold)
        stats = stats[stats["count"] >= topQuan].head(h)

    return stats[["movieId", "title", "genres", "count", "mean", "score"]]





def content_based(movie_title, h=15, w_content=0.8, w_pop=0.2, max_genre_frac=0.3):
    data = movies.copy()

    data["genres"] = data["genres"].fillna("").str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer()
    matrix_of_genres = vectorizer.fit_transform(data["genres"])

    found = data[data["title"].str.lower() == movie_title.lower()]
    if found.empty:
        raise ValueError("Movie was not found. Check the title.")

    movie_index = found.index[0]
    similarities = cosine_similarity(matrix_of_genres[movie_index], matrix_of_genres).flatten()
    data["similarity"] = similarities

    pop_map = popularity().set_index("movieId")["score"]
    data["popularity"] = data["movieId"].map(pop_map)
    data["popularity"] = data["popularity"].fillna(data["popularity"].median()) / 5
    data["score"] = w_content * data["similarity"] + w_pop * data["popularity"]

    ranked = (data.drop(index=movie_index).sort_values("score", ascending=False).head(h * 5))
    ranked = diversify(ranked, h, max_genre_frac)

    return ranked.head(h)[["title", "genres", "similarity", "popularity", "score"]]



def collab_user_based(user_id, h=15):
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

    watched = set(ratings.loc[ratings["userId"] == user_id, "movieId"])

    if user_id not in matrix.index or not watched:
        return popularity(h=h).rename(columns={"score": "rating"})[["title", "genres", "rating"]]

    similar = cosine_similarity(matrix)
    similar = pd.DataFrame(similar, index=matrix.index, columns=matrix.index)
    nearest_users = similar[user_id].drop(user_id).sort_values(ascending=False).head(20).index

    result = ratings[ratings["userId"].isin(nearest_users)].groupby("movieId")["rating"].agg(["mean", "count"])
    result = result[result["count"] >= 3]
    result = result["mean"].drop(index=list(watched), errors="ignore").sort_values(ascending=False).head(h)
    return result.reset_index().rename(columns={"mean": "rating"}).merge(movies, on="movieId")[["title", "genres", "rating"]]





def collab_item_based(movie_title, h=10):
    found = movies[movies["title"].str.lower() == movie_title.lower()]
    if found.empty:
        raise ValueError("Movie was not found. Check the title.")
    movie_id = found.iloc[0]["movieId"]

    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

    if movie_id not in matrix.columns:
        return content_based(movie_title, h=h)[["title", "genres", "similarity"]]

    similarities = cosine_similarity(matrix.T)
    similarities = pd.DataFrame(similarities, index=matrix.columns, columns=matrix.columns)
    result = similarities[movie_id].drop(movie_id).sort_values(ascending=False).head(h)
    result = result.reset_index(name="similarity").rename(columns={"index": "movieId"})
    return result.merge(movies, on="movieId")[["title", "genres", "similarity"]]


def hybrid(user_id, movie_title, h=10):
    content = content_based(movie_title, h=100).rename(columns={"similarity": "content_score"})
    item = collab_item_based(movie_title, h=100).rename(columns={"similarity": "item_score"})

    result = content.merge(item[["title", "item_score"]], on="title", how="outer")
    result["content_score"] = result["content_score"].fillna(0)
    result["item_score"] = result["item_score"].fillna(0)
    result["score"] = 0.5 * result["content_score"] + 0.5 * result["item_score"]

    watched_ids = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    watched_titles = set(movies.loc[movies["movieId"].isin(watched_ids), "title"])
    result = result[~result["title"].isin(watched_titles)]
    return result.sort_values("score", ascending=False).head(h)[["title", "genres", "content_score", "item_score", "score"]]




def same_genre(movie_title, h=10, w_genre=0.7, w_pop=0.3):
    found = movies[movies["title"].str.lower() == movie_title.lower()]
    if found.empty:
        raise ValueError("Movie was not found. Check the title.")
    source = found.iloc[0]
    source_genres = set(source["genres"].split("|"))
    result = movies[movies["movieId"] != source["movieId"]].copy()
    result["common_genres"] = result["genres"].apply(lambda genres: len(source_genres.intersection(genres.split("|"))))
    result = result[result["common_genres"] > 0]

    result["genre_score"] = result["common_genres"] / len(source_genres)

    pop_map = popularity().set_index("movieId")["score"]
    result["popularity"] = result["movieId"].map(pop_map)
    result["popularity"] = result["popularity"].fillna(result["popularity"].median()) / 5

    result["score"] = w_genre * result["genre_score"] + w_pop * result["popularity"]

    return result.sort_values("score", ascending=False).head(h)[["title", "genres", "common_genres", "popularity", "score"]]


if __name__ == "__main__":
    movie = "Toy Story (1995)"
    user = 1

    print("POPULARITY (heuristic)")
    print(popularity(h=15))

    print(f"\nCONTENT-BASED: {movie}")
    print(content_based(movie))

    print(f"\nUSER-BASED user {user}")
    print(collab_user_based(user))

    print(f"\nUSER-BASED (cold user 3463424234)")
    print(collab_user_based(3463424234))

    print(f"\nITEM-BASED: {movie}")
    print(collab_item_based(movie))

    print(f"\nHYBRID: user {user}, {movie}")
    print(hybrid(user, movie))

    print(f"\nSAME GENRE: {movie}")
    print(same_genre(movie))
