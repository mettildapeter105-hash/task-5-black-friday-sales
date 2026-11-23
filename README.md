# Task 5 - Black Friday Sales Prediction

Machine learning project that predicts the purchase amount of Black Friday transactions from customer, city and product information. It is a regression problem, evaluated with RMSE.

## Dataset
`BlackFridaySales.csv` - 550,068 transactions from a Black Friday sale (Analytics Vidhya hackathon data), with 5,891 customers and 3,631 products.

| Column | Meaning |
|---|---|
| User_ID, Product_ID | customer and product identifiers |
| Gender, Age, Occupation | customer details |
| City_Category, Stay_In_Current_City_Years | city information |
| Marital_Status | 0 = single, 1 = married |
| Product_Category_1/2/3 | product categories (2 and 3 are often missing) |
| **Purchase** | amount spent (target) |

The dataset is about 25 MB, so it is not included in this repository. Download it from the Slash Mark task page and put `BlackFridaySales.csv` next to the script (or in a `Data/` folder), or pass its path with `--data`.

## Approach
1. **Explore** the data (`eda_plots.png`): men, ages 51-55 and city C customers spend the most on average.
2. **Clean and encode**: gender and age encoded numerically, city one-hot encoded, "4+" years converted to 4, missing Product_Category_2/3 filled with 0 (no extra category).
3. **Feature engineering** (the biggest improvement):
   - how often each customer and each product appears in the data
   - the average purchase of each customer and product (target encoding), calculated **out-of-fold on the training set only**, so no test information leaks into the features
   - the number of categories a product belongs to
4. **Split** 80% train / 20% test (440,054 / 110,014 rows).
5. **Compare five models**: baseline (predict the mean), Linear Regression, Random Forest, Hist Gradient Boosting and XGBoost.

## Setup and run
```bash
pip install -r requirements.txt
python black_friday_sales.py
python black_friday_sales.py --data path/to/BlackFridaySales.csv   # custom dataset location
python black_friday_sales.py --sample 100000                       # quick run on a subset
python black_friday_sales.py --no-xgboost                          # without XGBoost
```
Results and plots are saved in `outputs/`.

## Results (20% test set)
| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Baseline (predict mean) | 5012.6 | 4060.3 | 0.0000 |
| Linear Regression | 2610.4 | 1940.8 | 0.7288 |
| Random Forest | 2548.2 | 1875.6 | 0.7416 |
| Hist Gradient Boosting | 2520.8 | 1856.3 | 0.7471 |
| **XGBoost** | **2514.8** | **1849.4** | **0.7483** |

**Best model: XGBoost, RMSE 2,514.8** - about 49.8% lower than the baseline, explaining about 75% of the variation in purchase amounts.

The most important features are the product's average purchase amount, the product category and the customer's average purchase amount: what is bought matters far more than who buys it (age, gender and city have very little influence).

## Result images
- `eda_plots.png` - purchase distribution and average purchase by customer group
- `model_comparison.png` - test RMSE of each model
- `predicted_vs_actual.png` - predictions against the real purchase amounts
- `feature_importance.png` - which features matter most

## Project structure
```
black_friday_sales.py
requirements.txt
eda_plots.png
model_comparison.png
predicted_vs_actual.png
feature_importance.png
README.md
```
