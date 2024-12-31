# Behind Netflix's Recommendations: The Influence of Singular Value Decomposition (SVD)

# Movie Recommendation System

This is a movie recommendation system that uses collaborative filtering with Singular Value Decomposition (SVD) to generate personalized movie recommendations based on user ratings.

## Overview

The system implements a recommendation engine using the following components:
- Data loading and processing from Netflix Prize dataset
- SVD model training using the Surprise library
- Cross-validation for model evaluation
- Recommendation generation with confidence scoring
- Visualization of model performance metrics

## Prerequisites

### Dependencies
```bash
pip install pandas numpy scikit-surprise matplotlib
```

### Required Libraries
- pandas
- numpy
- surprise
- matplotlib.pyplot

# Dataset Download

## 1. Download Dataset from Kaggle

1. Go to [Netflix Prize Data on Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/data)
2. If you're not logged in, you'll need to create a Kaggle account or sign in
3. Click the "Download" button on the dataset page
4. The download will be a zip file named `archive.zip`

## 2. Setup Data Directory

1. Create a `data` directory in your project folder:
```bash
mkdir data
```

2. Extract the downloaded `archive.zip` file
3. Move the following files to your `data` directory:
   - `combined_data_1.txt`
   - `combined_data_2.txt`
   - `combined_data_3.txt`
   - `combined_data_4.txt`
   - `movie_titles.csv`

## 3. Verify Dataset Structure

### Check movie_titles.csv format
The file should look like this:
```
1,2003,Dinosaur Planet
2,2004,Isle of Man TT 2004
3,2003,Character
...
```

### Check combined_data files format
Each file should contain entries like this:
```
1:
2305047,4,2005-06-06
1286514,4,2005-06-06
2059652,4,2005-06-06
1950305,5,2005-06-06
2:
1395461,3,2005-06-06
2scape to this structure
...
```

## Usage

### Basic Usage

1. Place your data files in a directory named `data`
2. Run the script:
```bash
python movie_recommender.py
```

### Key Functions

1. `load_and_process_data(path, sample_size=None)`
   - Loads and processes the movie and rating data
   - Optional sampling for testing with smaller datasets
   - Returns processed ratings and movie title dataframes

2. `train_svd_model(df_ratings, n_factors=100, cv_folds=5)`
   - Trains the SVD model with specified parameters
   - Performs cross-validation
   - Returns trained model and validation results

3. `get_recommendations(svd_model, user_id, df_ratings, df_mov_titles, top_n=20, min_ratings=5)`
   - Generates personalized recommendations for a specific user
   - Returns top N recommendations based on predicted ratings and popularity

4. `plot_rmse(results)`
   - Visualizes model performance metrics (RMSE and MAE)

5. `save_recommendations_to_csv(recommendations, user_id)`
   - Saves recommendations to a CSV file

## Customization

You can modify the following parameters in the code:
- `sample_size`: Number of ratings to use (default: 2,000,000)
- `n_factors`: Number of latent factors in SVD (default: 100)
- `cv_folds`: Number of cross-validation folds (default: 5)
- `top_n`: Number of recommendations to generate (default: 20)
- `min_ratings`: Minimum number of ratings for movie consideration (default: 5)

## Output

The system generates:
1. Model performance metrics (RMSE and MAE)
2. Visualization of cross-validation results
3. Personalized movie recommendations for specified users
4. CSV files containing recommendations for each user

## Example Output Format

Recommendations are saved in CSV files with the following columns:
- Movie_Id
- Predicted_Rating
- Name
- Year
- Number_of_Ratings
- Score

## Notes

- The system uses stratified sampling to maintain user representation when working with smaller datasets
- Confidence scoring combines predicted ratings with movie popularity
- Users with fewer than 5 ratings will receive a warning about potential recommendation accuracy

## Performance Considerations

- Processing the full Netflix dataset requires significant memory
- Using the `sample_size` parameter is recommended for testing
- The script uses all available CPU cores for cross-validation (`n_jobs=-1`)