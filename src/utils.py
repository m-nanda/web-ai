import pandas as pd
import numpy as np
from collections import OrderedDict
import sklearn
import warnings, random
warnings.filterwarnings("ignore")

CUST_NAME_SAMPLES = ["Billie", "Conley", "Margot", "Conrad", "Harlow"]
IMG_PATH = "img/"

def add_synthetic_data(df: pd.DataFrame, ll: float = 0.8, ul: float = 1.2) -> pd.DataFrame:
    """
    Adds synthetic data to a DataFrame by replicating rows, generating random percentages,
    and scaling values based on specified limits.

    Args:
        df: The input DataFrame.
        ll: The lower limit for the uniform random distribution (default: 0.8).
        ul: The upper limit for the uniform random distribution (default: 1.2).

    Returns:
        A new DataFrame with synthetic data added.

    Raises:
        ValueError: If the input DataFrame is empty or if any column contains non-numeric data.
    """

    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' contains non-numeric data.")

    # Replicate each row
    df_replicated = df.loc[df.index.repeat(1)]

    # Generate random percentage for each row
    random_percentages = [np.random.uniform(low=ll, high=ul)]

    # Calculate new values for each column
    for col in df_replicated.columns:
        df_replicated[col] *= random_percentages

    # Adjust "ratings" values max at 5
    df_replicated["ratings"] = df_replicated["ratings"].apply(lambda x: 5.00 if x>5 else round(x,2) )

    # Adjust "ratings" values max at 4
    df_replicated["purchase_date"] = df_replicated["purchase_date"].apply(lambda x: 4 if x>4 else round(x) )
    df_replicated["category"] = df_replicated["category"].apply(lambda x: 3 if x>3 else round(x) )

    # Shift "product_id" values by 200 (a constant)
    df_replicated["product_id"] += 200
    
    # Round numeric columns
    for col in ["customer_id", "product_id", "page_views", "time_spent", "price", "category", "purchase_date"]:
        df_replicated[col] = df_replicated[col].round()
        df_replicated[col] = df_replicated[col].astype(int)

    return df_replicated

def generate_new_ids(df: pd.DataFrame, col: str, offset: int = 10) -> pd.DataFrame:
    """
    Generates new unique IDs for a specific column in a DataFrame.

    Args:
        df: The input DataFrame.
        col: The column name to generate new IDs for.
        offset: The offset to add to the maximum ID value (default: 10).

    Returns:
        The input DataFrame with updated IDs.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
    """

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

    max_id = df[col].max() + offset
    new_ids = list(range(max_id + 1, max_id + df.shape[0] + 1))
    random.shuffle(new_ids)
    df[col] = new_ids
    return df

def get_recommendation(
        df: pd.DataFrame, 
        user_features: pd.DataFrame,
        bought_items: pd.DataFrame,
        model: sklearn.neighbors._classification.KNeighborsClassifier,
        user_idx: dict,  
        uid: int, 
        item_idx: dict, 
        n: int = 5, 
        k: int = 7
    ) -> tuple[list[int], list[int]]:
    """
    Recommends items to a user based on collaborative filtering.

    Args:
        df (pd.DataFrame): Base Data
        user_features (pd.DataFrame): user features that are `page_views` and `time_spent`
        bought_items (pd.DataFrame): historical data on product purchases by users
        model (sklearn.neighbors._classification.KNeighborsClassifier): 
        user_idx (dict): user index to look up
        uid (int): The user ID for whom to generate recommendations.
        item_idx (dict): item index to look up
        n (int, optional): The number of recommendations to return. Defaults to 5.
        k (int, optional): The number of nearest neighbors. Defaults to 8.

    Returns:
        tuple[list[int], list[int]]: A tuple containing two lists:
            - The list of top k most similar user IDs.
            - The list of k recommended item IDs.

    Raises:
        ValueError: If the user ID is not found in the user features data.
    """

    if uid not in user_features.index:
        raise ValueError(f"User with ID {uid} not found in user features data.")

    # Prepare input: extract scaled user features for the given user
    input_recsys = user_features.loc[uid].values

    # Find similar users based on the user features using k-nearest neighbors
    similar_users = model.kneighbors([input_recsys], n_neighbors=k+1, return_distance=False)[0]

    # Recommend items based on frequently bought items by similar users
    recommend_items = []
    for idx in similar_users[1:]:
        # Find items bought by the similar user (excluding the user themself)
        tmp_item = bought_items.iloc[[idx]].sum(axis=0).sort_values(ascending=False)
        # Filter out items not bought by the user and keep only the top k recommendations
        tmp_item = list(tmp_item[tmp_item > 0].index)#[:n]
        recommend_items += tmp_item
    
    # remove already bought item
    ## Sample case: customer id 5 already bought item_id 101
    already_bought = df.loc[df['customer_id'] == uid, 'product_id'].tolist()  # Get already bought items directly
    recommend_items = [item for item in recommend_items if item not in already_bought]  # Filter in a single line

    # Use OrderedDict to remove duplicates and keep the top k recommendations
    if n <= len(recommend_items):
        recommend_items = list(OrderedDict.fromkeys(recommend_items))[:n]
    
    # get user ID
    similar_users = [user_idx[i] for i in similar_users]
    recommend_categories = [item_idx[i] for i in recommend_items]
    return similar_users, recommend_items, recommend_categories

def calculate_metrics(data:dict, k:int = 5) -> tuple[dict, dict]:
  """
  Calculates Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP) for recommendation data.

  Args:
    data: A dictionary where keys are user IDs and values are dictionaries containing
          'ground_truth' (list of relevant items) and 'recommendation' (list of predicted items).

  Returns:
    A dictionary containing calculated MRR and MAP values.
  """
  rr = 0.0
  ap = 0.0
  for user_id, user_data in data.items():
    ground_truth = user_data['ground_truth']
    recommendation = user_data['recommendation']

    # Calculate RR
    for i, item in enumerate(recommendation):
      if item in ground_truth:
        rr += 1 / (i + 1)
        break

    # Calculate AP
    relevant_at_k = recommendation[:k].count(ground_truth[0])
    precision_at_k = relevant_at_k / k
    ap += precision_at_k / len(ground_truth)

  # Calculate MRR and MAP
  mrr = rr / len(data)
  map = ap / len(data)

  return {'MRR': mrr, 'MAP': map}