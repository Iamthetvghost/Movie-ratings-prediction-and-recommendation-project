# MovieLens 32M: Rating Prediction and Movie Recommendation

## Overview

This project explores how statistical learning methods can be used to predict movie ratings and recommend movies based on user preferences. Using the MovieLens 32M dataset, I compare several rating prediction models and investigate how different representations of movies influence recommendation results.

The main research question is:

**How do different representations of movie relationships affect the recommendations produced for a given set of movies?**

The project focuses on two tasks:

- **Rating prediction:** Predicting how a user might rate a movie based on historical user–movie interactions.
- **Movie recommendation:** Comparing recommendations based on predefined genres, observed rating patterns, and latent factors learned by SVD.

## Dataset

This project uses the [MovieLens 32M dataset](https://grouplens.org/datasets/movielens/32m/), provided by GroupLens Research.

The dataset contains approximately:

- 32 million ratings
- 87,585 movies
- 200,948 users

It also includes movie genres, tags, and links to external movie databases.

The dataset is described in:

Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. https://doi.org/10.1145/2827872

The dataset is not included in this repository. Please download it from the official source and follow its terms of use.

## Methods

### 1. Rating Prediction

Four models are compared:

- **Linear Regression:** A baseline model for predicting ratings from user and movie identifiers.
- **Ridge Regression:** A regularised linear model intended to reduce overfitting.
- **User–Movie Bias Model:** A baseline that accounts for systematic differences between users and movies.
- **Singular Value Decomposition (SVD):** A latent-factor model that learns lower-dimensional representations of users and movies from observed ratings.

Models are evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (\(R^2\)).

In the experiments, SVD achieved better rating prediction performance than the simpler baseline models.

### 2. Movie Recommendation

Three approaches are compared using a set of input movies as preference signals.

- **Genre-based similarity:** Represents each movie using its predefined genre labels.
- **Rating-based similarity:** Represents each movie by its ratings from users.
- **SVD-based similarity:** Represents each movie using latent factors learned by the SVD rating prediction model.

Cosine similarity is used to rank candidate movies relative to the input movies.

The experiments show that the methods capture different aspects of movie relationships. Genre-based similarity is interpretable but limited by coarse categories. Rating-based similarity uses behavioural information but can be affected by sparse ratings and differences in popularity. SVD can reveal less obvious associations, although its latent factors are not directly interpretable.

Recommendation results are evaluated qualitatively by testing different combinations of input movies. Unlike rating prediction, recommendation quality is subjective, and a higher similarity score does not necessarily imply a better recommendation.

## Visualisation

The visualisations follow a consistent and minimalist style, with clear titles, labels, and legends to make the results easy to read and compare. A restrained colour palette is used to distinguish different models, variables, and groups while avoiding unnecessary visual complexity.

Different visualisation types are selected according to the purpose of each analysis. Heatmaps are used to display patterns in genre preferences, while tables and comparison plots are used to present model performance. Movie posters and titles are used to make recommendation results easier to recognise and compare.

The Jupyter notebook in the `notebook/` directory includes visualisations to help interpret model performance and learned movie representations. The figures are chosen according to their purpose, such as comparing prediction errors or examining the latent factors learned by SVD. The other Python file in the `notebook/` directory contains the code used to generate the files required by the web application.

The repository also includes a small interactive web application in the `website/` directory. The application allows users to enter three movie titles and receive ten recommendations based on cosine similarity between SVD-learned movie latent factors. It uses the pre-trained movie vectors exported from the notebook, so the model does not need to be retrained when the application runs.

The application is deployed online and can be accessed here: **[Film Affinity](https://cghostmovie-movies-recommendation.streamlit.app/)**.

## FAIR Principles

The project considers the four FAIR principles: Findable, Accessible, Interoperable, and Reusable.

### Findable

The repository has a descriptive title and a README that clearly explains the project's purpose, research questions, methods, and structure. This helps users identify the project and understand its contents. The MovieLens dataset is linked to its official source, and its associated publication is provided so that users can identify and locate the original dataset.

### Accessible

The source code and project documentation are available through the repository. The dataset can be obtained separately from the official GroupLens website. The interactive application is also deployed online, allowing users to access the recommendation system without running the notebook locally.

### Interoperable

The project uses commonly adopted tools and formats, including Python, Jupyter Notebook, CSV files, and standard Python libraries. These make the data and code easier to work with in other Python-based environments and workflows.

### Reusable

The README provides information about the dataset, methods, dependencies, and steps required to reproduce the analysis. Random seeds are specified where applicable to support reproducibility. The original project code is released under the MIT License, allowing others to reuse and adapt it under the terms of that license. The MovieLens dataset remains subject to its own terms of use.

## Reproducibility

The analysis is implemented in a Jupyter Notebook.

### Requirements

Python 3 and the packages listed in `requirements.txt` are required.

### Setup

1. Clone or download this repository.
2. Download the MovieLens 32M dataset from the [official website](https://grouplens.org/datasets/movielens/32m/).
3. Extract the dataset so that the files are available under `data/ml-32m/`.
4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Open the notebook in Jupyter and run the cells in order.

The notebook uses the dataset locally and does not require uploading the full dataset to the repository. Random seeds are specified where applicable to support reproducibility.

## Limitations and Future Work

The rating prediction models are evaluated quantitatively, but the recommendation methods are compared primarily through qualitative experiments. The selected input movies and the interpretation of recommendations are subjective, so the results should not be treated as a formal benchmark of recommendation quality.

Potential future work includes exploring neural-network-based recommendation methods and developing a lightweight interactive interface that allows users to enter movies and compare recommendations from all three approaches.

## License

The original code in this repository is released under the [MIT License](LICENSE).

The MovieLens dataset is provided by GroupLens Research and is subject to its own terms of use. The MIT License for this project's code does not apply to the dataset.
