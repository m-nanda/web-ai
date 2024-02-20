# Description  

In the competitive landscape of online retail, understanding customer behavior and predicting their next purchase is crucial for success. This project tackles this challenge by developing a web-based AI application specifically designed to unlock the power of historical data and predict the next product a customer is likely to buy. By harnessing the insights from past interactions, purchase history, and product details, this application aims to empower marketing teams with the ability to target customers more effectively and deliver personalized recommendations that resonate.

# Objectives  

- Build a data-driven recommendation engine: Train predictive model with utilizing user interaction, purchase, and product details; incorporating algorithms that capable to deliver accurate and dynamic product recommendations.
- Design a seamlessly integrated web interface: Provide a user-friendly web application to unlock valuable customer insights, & predict next product choices.

# Project Structure

This project is organized into several directories:

- `data/`:
    - This directory holds the initial (sample) data used for development and experimentation.
    - `customer_interactions.csv`: includes information about customer interactions on the website, such as the number of page views and time spent.
    - `product_details.csv`: contains records of customer purchases, including the product purchased and the date of purchase.
    - `purchase_hitory.csv`: provides details about each product, such as its category, price, and ratings.

- `img/`: 
    - This directory contains images that use in web-ui.

- `notebook/`:
    - This directory contains Jupyter notebooks used for data exploration and model development.
    - `experiment.ipynb`: This notebook provides a detailed analysis of the input data, including visualizations and insights into user behavior, product features, and potential correlations. This exploration helps identify patterns and choose appropriate features for the model.

- `src/`:
    - This directory holds custom libraries developed specifically for this project.
    - `utils.py`: This file contains helper functions used throughout the project, such as data preprocessing, feature engineering, and model evaluation. These functions streamline the development process and ensure consistent code quality.
    - `web.py`: This file contains the code responsible for building the web-based user interface (UI) of the recommendation application. This code enables users to interact with the model and receive product recommendations.

- `pipeline_objects.bin`: This file stores binary objects representing the trained K-Nearest Neighbors (KNN) model and its associated prediction pipeline. This pipeline allows for efficient and reliable deployment of the model in production.

- `README.md`: (This file) describes the project, its goals, and usage instructions. It provides a high-level overview of the project and serves as a reference guide for users.

- `requirements.txt`:This file lists all the external libraries and dependencies required to run the project. This ensures that anyone can easily replicate your environment and run the application.

# Usage  

## Prerequisites:

- Python >= 3.10 (https://www.python.org/)
- Git (https://git-scm.com/)

## Steps:  

**1. Local Setup:**

- **Clone the Repository:** Copy the project to your local machine using Git:

```
git clone https://github.com/m-nanda/web-ai.git
cd web-ai
```

- **Create and Activate Virtual Environment:** Establish an isolated environment for project dependencies:

```
python -m venv .venv
source .venv/bin/activate # On Linux/macOS
.venv\Scripts\activate # On Windows
```

- **Install Dependencies:** Download and install necessary libraries:

```
pip install -r requirements.txt
```

**2. Development:**

- **Experimentation:** Explore the model development process in the Jupyter Notebook `notebook/experiment.ipynb`. It walks you through data exploration, model implementation, evaluation, and detailed explanations and ideas for further improvement.

**3. Production:**

- **Run the Web Application:** Launch the user-friendly interface for marketing teams:

```
streamlit run src/web.py
```

This command opens the application in your web browser, accessible for product recommendation predictions.


# Model Development  

## Steps:  
- Preparation: Load libraries, prepare data (exploration, cleaning, upsampling).  
- Modeling: Main model was KNN, with two baseline comparison:
    - Random recommendation model serves as a basic performance benchmark.  
    - Highest rating model represents a simple popularity-based approach.  
- Evaluation: Assess performance using MRR and potentially MAP metrics.  
- Pipeline: Save the model and prediction pipeline for deployment.  

## Key Considerations:
- Initial data distribution is sparse, provide opportunities for upsampling/augmentation.
- Upsampling/augmentation involves replicating data due to very limited initial data.
- Correlation patterns between user-item-category features exist in each product category.
- KNN is chosen due simple and suitable for initial development with data conditions.

## Results:
- KNN can works with the data and outperforms baseline methods in all metrics.
- Recommendation metrics is promising and aligns with initial plan.