import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt

def load_and_process_data(path, sample_size=None):
    # Load movie catalog
    df_mov_titles = pd.read_csv(path + '/movie_titles.csv', 
                               sep=',', 
                               header=None, 
                               names=['Movie_Id', 'Year', 'Name'],
                               usecols=[0, 1, 2], 
                               encoding="ISO-8859-1")
    df_mov_titles.set_index('Movie_Id', inplace=True)

    # Load ratings files
    files = ['combined_data_1.txt', 'combined_data_2.txt', 
             'combined_data_3.txt', 'combined_data_4.txt']
    df_ratings = pd.DataFrame()
    
    # Only load first file for testing if needed
    if sample_size:
        files = files[:1]

    for f in files:
        staging = pd.read_csv(path + '/' + f, 
                            header=None, 
                            names=['Cust_Id', 'Rating'], 
                            usecols=[0, 1])
        staging['Rating'] = staging['Rating'].astype(float)
        
        if len(df_ratings) > 0:
            df_ratings = pd.concat([df_ratings, staging], ignore_index=True)
        else:
            df_ratings = staging
        del staging

    # Process movie IDs
    movies_IDs = pd.DataFrame(pd.isnull(df_ratings.Rating))
    movies_IDs = movies_IDs[movies_IDs['Rating'] == True].reset_index()

    movies_IDs_fin = []
    mo = 1
    for i, j in zip(movies_IDs['index'][1:], movies_IDs['index'][:-1]):
        temp = np.full((1, i - j - 1), mo)
        movies_IDs_fin = np.append(movies_IDs_fin, temp)
        mo += 1

    last_ = np.full((1, len(df_ratings) - movies_IDs.iloc[-1, 0] - 1), mo)
    movies_IDs_fin = np.append(movies_IDs_fin, last_)

    # Clean and prepare final dataframe
    df_ratings = df_ratings[pd.notnull(df_ratings.Rating)]
    df_ratings['Movie_Id'] = movies_IDs_fin.astype(int)
    df_ratings['Cust_Id'] = df_ratings['Cust_Id'].astype(int)

    # Sample if requested
    if sample_size:
        # Stratified sampling to maintain user representation
        user_counts = df_ratings['Cust_Id'].value_counts()
        users_with_min_ratings = user_counts[user_counts >= 5].index
        df_ratings = df_ratings[df_ratings['Cust_Id'].isin(users_with_min_ratings)]
        df_ratings = df_ratings.sample(n=min(sample_size, len(df_ratings)), random_state=42)

    return df_ratings, df_mov_titles

def train_svd_model(df_ratings, n_factors=100, cv_folds=5):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_ratings[['Cust_Id', 'Movie_Id', 'Rating']], reader)
    
    # Train model with improved parameters
    svd = SVD(
        n_factors=n_factors,
        biased=True,  # Enable biased learning
        lr_all=0.005,  # Learning rate for all parameters
        reg_all=0.02,  # Regularization for all parameters
        n_epochs=30,   # Number of epochs
        random_state=42
    )
    
    results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=cv_folds, n_jobs=-1)
    
    # Train final model on full dataset
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    
    return svd, results

def get_recommendations(svd_model, user_id, df_ratings, df_mov_titles, top_n=20, min_ratings=5):
    # Get movies not rated by user
    user_ratings = df_ratings[df_ratings['Cust_Id'] == user_id]
    
    # Check if user has minimum number of ratings
    if len(user_ratings) < min_ratings:
        print(f"Warning: User {user_id} has less than {min_ratings} ratings. Recommendations may be less accurate.")
    
    # Get movies with sufficient ratings
    movie_ratings = df_ratings['Movie_Id'].value_counts()
    popular_movies = movie_ratings[movie_ratings >= min_ratings].index
    
    # Generate predictions only for movies with sufficient ratings
    predictions = []
    for movie_id in popular_movies:
        if movie_id not in user_ratings['Movie_Id'].values:
            pred = svd_model.predict(user_id, movie_id).est
            predictions.append([movie_id, pred])
    
    # Create recommendations dataframe
    recs_df = pd.DataFrame(predictions, columns=['Movie_Id', 'Predicted_Rating'])
    recs_df = recs_df.merge(df_mov_titles[['Name', 'Year']], 
                           left_on='Movie_Id', 
                           right_index=True)
    
    # Add confidence score based on number of ratings
    movie_rating_counts = df_ratings['Movie_Id'].value_counts()
    recs_df['Number_of_Ratings'] = recs_df['Movie_Id'].map(movie_rating_counts)
    
    # Sort by predicted rating and number of ratings
    recs_df['Score'] = recs_df['Predicted_Rating'] * np.log1p(recs_df['Number_of_Ratings'])
    
    return recs_df.sort_values('Score', ascending=False).head(top_n)

def plot_rmse(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot RMSE
    ax1.plot(results['test_rmse'], label='RMSE per fold')
    ax1.axhline(y=results['test_rmse'].mean(), color='r', linestyle='--', label='Average RMSE')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE per Fold')
    ax1.legend()
    
    # Plot MAE
    ax2.plot(results['test_mae'], label='MAE per fold')
    ax2.axhline(y=results['test_mae'].mean(), color='r', linestyle='--', label='Average MAE')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE per Fold')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def save_recommendations_to_csv(recommendations, user_id):
    recommendations.to_csv(f'recommendations_for_user_{user_id}.csv', index=False)
    print(f"Recommendations saved to 'recommendations_for_user_{user_id}.csv'.")

# Usage 
if __name__ == "__main__":
    # Load larger sample of data for better model training
    df_ratings, df_mov_titles = load_and_process_data('data', sample_size=2000000)
    print(f"Working with {len(df_ratings)} ratings")
    
    # Train improved model
    svd_model, results = train_svd_model(df_ratings)
    print("\nCross-validation results:")
    print(f"Average RMSE: {results['test_rmse'].mean():.4f}")
    print(f"Average MAE: {results['test_mae'].mean():.4f}")
    
    # Visualize model performance
    plot_rmse(results)
    
    # Get recommendations for a user
    user_ids = [1765963, 1462327]
    for user_id in user_ids:
        recommendations = get_recommendations(svd_model, user_id, df_ratings, df_mov_titles)
        print(f"\nRecommendations for user {user_id}:")
        print(recommendations[['Name', 'Year', 'Predicted_Rating', 'Number_of_Ratings']])
        save_recommendations_to_csv(recommendations, user_id)