import streamlit as st
import pandas as pd
from utils import *
import warnings, random, pickle, math
warnings.filterwarnings("ignore")

def main() -> None:
    """
    Main function for the Terra Store web app.

    Loads the pipeline objects, displays the landing page, handles customer ID input,
    generates and displays recommendations, and showcases top-rated and all items.
    """

    # Load pipeline objects 
    pipeline_objects = load_pipeline()
    db: pd.DataFrame = pipeline_objects["database"]
    stored_features: pd.DataFrame = pipeline_objects["scaled_user_features"]
    bought_items: pd.DataFrame = pipeline_objects["bought_items"]
    model: object = pipeline_objects["model"]
    user_idx: dict = pipeline_objects["user_idx"]
    item_idx: dict = pipeline_objects["item_idx"]

    # Landing page with hero image and description (replace with your image)
    st.title("Welcome to Terra Store! 🛍️")
    st.subheader("The future of shopping is here: AI-powered recommendations")
    st.markdown("""
        Shopping made simple, personalized, and delightful. That's the Terra Store difference.
        Become a member today and experience the magic of AI-powered shopping!
    """)

    # Customer ID input and validation
    customer_id = st.text_input("Already have Customer ID? Please enter here")
    if len(customer_id) > 0:
        try:
            customer_id: int = int(customer_id)
        except ValueError:
            st.error("Please enter a valid ID (integer)")
            customer_id = None

    # Recommendation generation and display
    if customer_id in db.customer_id:
        try:
            similar_user, product_recommendations, category_item = get_recommendation(
                db, stored_features, bought_items, model, user_idx, customer_id, item_idx
            )
            cust_name = random.choice(CUST_NAME_SAMPLES) 
            st.write("---")
            st.subheader(f"Hi _{cust_name}_ ! We have some recommendations for you")
            show_recommendation(db, product_recommendations)
        except Exception:
            st.warning("It looks like you are not yet registered as our member")
            customer_id = None

    # Top-rated products
    st.write("---")
    top_ratings = db.sort_values("ratings", ascending=False).nlargest(n=5, columns="ratings")["product_id"]
    st.subheader(f"New to Terra Store? Here are most favorite items 😉")
    show_recommendation(db, top_ratings)

    # All products
    st.write("---")
    current_products = db["product_id"].unique()
    random.shuffle(current_products)
    st.subheader(f"Want to explore yourself?")
    show_recommendation(db, current_products)

def show_recommendation(
    db: pd.DataFrame,
    recommendations: list[str],
    n: int = 5,
) -> None:
    """
    Displays personalized product recommendations in a grid layout using Streamlit.

    Args:
        db (pd.DataFrame): DataFrame containing product data (product_id, category, ratings, price).
        recommendations (list[str]): List of recommended product IDs.
        n (int, optional): Maximum number of recommendations to display. Defaults to 5.
    """
    total_rows = math.ceil(n / 5)
    for row_index in range(total_rows):
        start_index = row_index * 5
        end_index = min(start_index + 5, len(recommendations))
        current_recommendations = recommendations[start_index:end_index]

        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]

        for col, product_id in zip(cols, current_recommendations):
            with col:
                # Fetch product details from DataFrame
                product_data = db[db["product_id"] == product_id]
                category = product_data["category"].values[0]
                rating = product_data["ratings"].values[0]
                price = product_data["price"].values[0]
                product_name = f"{product_id} - {category}"

                # Load and display product image
                st.image(f"{IMG_PATH}{product_id}.png")

                # Display product information
                st.write(f"{product_name} - {category} (⭐ {rating})")
                st.write(f"$ {price}")
                with st.expander("Item Description"):
                    st.write(f"{product_name}, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras tincidunt nulla elementum mauris convallis, sit amet condimentum erat viverra. Aenean.")

def load_pipeline() -> object:
    """
    Loads the pre-trained recommendation pipeline object.

    Returns:
        object: The loaded recommendation pipeline object.
    """
    with open("pipeline_objects.bin", "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object

if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="Terra Store",
        page_icon="🛍️🛒",
        layout="wide",
    )  

    # run the web
    main()