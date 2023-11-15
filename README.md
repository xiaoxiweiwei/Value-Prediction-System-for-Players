# Team Members
- Shuying Huang (andrew id: shuyingh)
- Xiaoxi Wei (andrew id: xiaoxiw)

# Data Source
Our project uses the following dataset:
[FIFA 22 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)  
We only use all Men Player Data across all years to conduct the projects. The **data** folder contains all data we use.

# Dataset Schema

### Added columns
- `unique_id`: 
  - Type: Long
  - Nullable: false
  - Description: A unique identifier for each player record. This is the primary key.
  - Constraints: Must be unique for each player record (primary key). Not null.

- `year`: 
  - Type: Integer
  - Nullable: false
  - Description: The year the player data pertains to.
  - Constraints: Cannot be null. Must be a valid year between 15 - 22.

### General Features

- `sofifa_id`: 
  - Type: Integer
  - Nullable: true
  - Description: The unique identifier for each player in the SoFIFA database.

- `player_url`: 
  - Type: String
  - Nullable: true
  - Description: The URL leading to the player's profile page on the SoFIFA website.

- `short_name`: 
  - Type: String
  - Nullable: true
  - Description: The player's short name.

- `long_name`: 
  - Type: String
  - Nullable: true
  - Description: The player's full name.

- `age`: 
  - Type: Integer
  - Nullable: true
  - Description: The player's age.

- `dob`: 
  - Type: Date
  - Nullable: true
  - Description: The player's date of birth.

- `height_cm`: 
  - Type: Integer
  - Nullable: true
  - Description: The player's height in centimeters.

- `weight_kg`: 
  - Type: Integer
  - Nullable: true
  - Description: The player's weight in kilograms.

- `nationality_id`: 
  - Type: Integer
  - Nullable: true
  - Description: The identifier for the player's national team.

- `nationality_name`: 
  - Type: String
  - Nullable: true
  - Description: The name of the player's nationality.

- `preferred_foot`: 
  - Type: String
  - Nullable: true
  - Description: The player's preferred foot (left/right).

- `real_face`: 
  - Type: String
  - Nullable: true
  - Description: Indicates if the player has a real face scan in the game (yes/no).

- `player_face_url`: 
  - Type: String
  - Nullable: true
  - Description: The URL of the player's face image.

### Performance Features

- `overall`: 
  - Type: Integer
  - Nullable: true
  - Description: The player's overall rating.

- `potential`: 
  - Type: Integer
  - Nullable: true
  - Description: The player's potential rating.

- `value_eur`: 
  - Type: Double
  - Nullable: true
  - Description: The player's market value.

- `wage_eur`: 
  - Type: Double
  - Nullable: true
  - Description: The player's weekly wage.

- `release_clause_eur`: 
  - Type: String
  - Nullable: true
  - Description: The player's release clause amount.

- `international_reputation`: 
  - Type: Integer
  - Nullable: true
  - Description: The player's international reputation on a 1-5 scale.
  - Constraints: Value must be an integer between 1 and 5 (inclusive).

- `weak_foot`: 
  - Type: Integer
  - Nullable: true
  - Description: The player's weak foot proficiency on a 1-5 scale.
  - Constraints: Value must be an integer between 1 and 5 (inclusive).

- `skill_moves`: 
  - Type: Integer
  - Nullable: true
  - Description: The number of skill moves the player is capable of.

- `work_rate`: 
  - Type: String
  - Nullable: true
  - Description: The player's work rate (e.g., High/ Low, Medium/ High).

- `body_type`: 
  - Type: String
  - Nullable: true
  - Description: The player's body type (e.g., Stocky, Normal, Lean).

- `player_positions`: 
  - Type: String
  - Nullable: true
  - Description: The positions the player is capable of playing in.

### Skill Features

- `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`:
  - Type: Integer
  - Nullable: true
  - Description: The player's core skill features.

- `attacking_crossing`, `attacking_finishing`, `attacking_heading_accuracy`, `attacking_short_passing`, `attacking_volleys`, `skill_dribbling`, `skill_curve`, `skill_fk_accuracy`, `skill_long_passing`, `skill_ball_control`:
  - Type: Integer
  - Nullable: true
  - Description: The player's detailed attacking and skill features.

- `movement_acceleration`, `movement_sprint_speed`, `movement_agility`, `movement_reactions`, `movement_balance`:
  - Type: Integer
  - Nullable: true
  - Description: The player's movement skill features.

- `power_shot_power`, `power_jumping`, `power_stamina`, `power_strength`, `power_long_shots`:
  - Type: Integer
  - Nullable: true
  - Description: The player's power features.

- `defending_marking_awareness`, `defending_standing_tackle`, `defending_sliding_tackle`:
  - Type: Integer
  - Nullable: true
  - Description:The player's defending skill features.

### Others  

- `mentality_aggression`, `mentality_interceptions`, `mentality_positioning`, `mentality_vision`, `mentality_penalties`, `mentality_composure`:
  - Type: Integer
  - Nullable: true
  - Description: These relate to mental aspects of the players.  
  
- `goalkeeping_diving`, `goalkeeping_handling`, `goalkeeping_kicking`, `goalkeeping_positioning`, `goalkeeping_reflexes`, `goal`
  - Type: Integer
  - Nullable: true
  - Description:These relate to goal aspects of the players.

# Machine Learning Model for Player Value Prediction

## Project Overview

This project aims to predict the overall value of each player based on their skill sets using machine learning models. We build two versions of the model: one in Apache Spark and the other in PyTorch. For each framework, two different classifiers/regressors are used. The selected models are then tuned for optimal performance and tested for accuracy.

## Dataset

(Briefly describe the dataset used, including its source, size, and main features.)

## Feature Engineering

### 1. Feature Type Casting

- Conversion of skill feature columns to numeric values after extracting them from string types.
- Handling of non-numeric values and errors.

### 2. Column Dropping

- Filling missing values in specified columns with their mean values.

### 3. Vector Assembling

- Consolidation of feature columns for model input.

### 4. Feature Scaling

- Application of StandardScaler for normalizing feature values.

### 5. PyTorch Tensor Conversion

- Conversion of scaled feature data to PyTorch tensors for model input.

## Models

### Linear Regression

#### Reasons for Selection

- Linear regression is a fundamental and widely used method for regression problems, providing a baseline for performance comparison.
- Its simplicity allows for a clear understanding of how input features affect the predicted player value.

### GBT and NN

- The neural network's ability to model complex, non-linear relationships makes it suitable for predicting player values, which may depend on intricate interactions between different skillsets.
- The flexibility of the architecture allows for easy adjustments and tuning to suit the specific nature of the dataset.
- By comparing this model's performance with that of linear regression, we can gain insights into the complexity and non-linearity of the data.

## Spark Version

### 1. Linear Regression Model

The Spark version of our project includes a Linear Regression Model implemented using PySpark's MLlib. This model aims to predict player values based on their skillsets using linear regression techniques.

#### Model Implementation

- The model is created using the `LinearRegression` class from PySpark MLlib.
- Features are set using `featuresCol`, and the label column is specified as 'overall'.

#### Hyperparameter Tuning

We employed a grid search method over a combination of different values for `regParam` and `maxIter`. The specific values tested were:

- `regParam`: 0.01, 0.1, 1.0
- `maxIter`: 10, 30, 50

#### Results

The results of the hyperparameter tuning are as follows:

| regParam | maxIter | Validation RMSE |
|----------|---------|-----------------|
| 0.01     | 10      | 2.793649927503585 |
| 0.01     | 30      | 2.793649927503586 |
| 0.01     | 50      | 2.793649927503613 |
| 0.1      | 10      | 2.805415603948817 |
| 0.1      | 30      | 2.8054156039488216 |
| 0.1      | 50      | 2.8054156039488096 |
| 1.0      | 10      | 2.913144924746838 |
| 1.0      | 30      | 2.913144924746848 |
| 1.0      | 50      | 2.913144924746848 |

#### Optimal Parameters

The best model performance was achieved with the following parameters:

- **Best `regParam`**: 0.01
- **Best `maxIter`**: 10
- **Best Validation RMSE**: 2.793649927503585

#### Analysis

- The lowest RMSE is observed with a `regParam` of 0.01 and `maxIter` of 10. This suggests that a lower level of regularization combined with a moderate number of iterations is optimal for our dataset.
- Increasing the `maxIter` beyond 10 did not significantly improve the RMSE for `regParam` of 0.01.
- Higher values of `regParam` (0.1 and 1.0) resulted in a slight increase in RMSE, indicating that too much regularization may not be beneficial for this specific model and data.

### 2. Gradient-Boosted Tree (GBT)

In addition to linear regression, we also implement a Gradient-Boosted Tree (GBT) model, which is a more complex approach suitable for capturing non-linear relationships in the data.

#### Model Implementation

- The GBT model is implemented using the `GBTRegressor` class from PySpark MLlib.
- Similar to the linear regression model, `featuresCol` and `labelCol` are set accordingly.

#### Hyperparameter Tuning

A grid search approach was employed to experiment with different combinations of `maxDepth` and `maxIter`. The tested ranges were:

- `maxDepth`: 4, 5, 6
- `maxIter`: 10, 20, 30

#### Results

The outcomes of this tuning process are as follows:

| maxDepth | maxIter | Validation RMSE |
|----------|---------|-----------------|
| 4        | 10      | 2.422824017719923 |
| 4        | 20      | 2.0858397109506788 |
| 4        | 30      | 1.8638631985296856 |
| 5        | 10      | 2.038985555619093 |
| 5        | 20      | 1.7765445213080413 |
| 5        | 30      | 1.636633216726328 |
| 6        | 10      | 1.7959977182481401 |
| 6        | 20      | 1.572301602247047 |
| 6        | 30      | 1.4361315745160284 |

#### Optimal Parameters

The best performance for the GBT model was observed with the following parameters:

- **Best `maxDepth`**: 6
- **Best `maxIter`**: 30
- **Best Validation RMSE**: 1.4361315745160284

#### Analysis

- The optimal results were obtained with a `maxDepth` of 6 and `maxIter` of 30, suggesting that a deeper tree and more iterations were beneficial for our dataset.
- Incremental increases in `maxIter` consistently improved the RMSE, particularly at higher tree depths.
- A depth of 6 seems to provide a good balance between model complexity and performance, as indicated by the lowest RMSE achieved.

### Model Evaluation

- The GBT model's performance is also evaluated using RMSE, allowing for a consistent comparison with the linear regression model.

#### Results

- The best model parameters (`maxDepth`, `maxIter`) and the RMSE score for the GBT model are reported, highlighting the model's performance on the validation dataset.

## PyTorch Version

### Linear Regression Model

The PyTorch version includes a Linear Regression Model, implemented using the PyTorch framework. This model is defined as a class `LinearRegressionModel` inheriting from `nn.Module`. The model consists of a single linear layer with an input size equal to the number of features in the dataset. The forward pass computes a linear transformation of the input features.

#### Model Architecture

- The model has a simple architecture with one linear layer (`nn.Linear`), making it suitable for understanding the linear relationships in the data.
- `input_size` is dynamically set based on the number of features, ensuring flexibility for different datasets.

#### Training Process

- The model is trained using a grid search approach to find the optimal regularization parameter and the number of iterations.
- The Root Mean Squared Error (RMSE) is used as the loss function, which is a standard choice for regression problems.
- Stochastic Gradient Descent (SGD) is employed as the optimizer.

### Hyperparameter Tuning

We used a grid search method to explore combinations of different values for `regParam` and `maxIter`. The values tested were:

- `regParam`: 0.01, 0.1, 1.0
- `maxIter`: 10, 30, 50

#### Results

The tuning results are as follows:

| regParam | maxIter | Validation RMSE |
|----------|---------|-----------------|
| 0.01     | 10      | 7.052768276618408 |
| 0.01     | 30      | 7.052531012543687 |
| 0.01     | 50      | 7.052517952145757 |
| 0.1      | 10      | 7.052557768048467 |
| 0.1      | 30      | 7.053355112806097 |
| 0.1      | 50      | 7.052695629832981 |
| 1.0      | 10      | 9.337242136130461 |
| 1.0      | 30      | 16.89485637561695 |
| 1.0      | 50      | 28.164455955092972 |

#### Optimal Parameters

The best performance was achieved with the following settings:

- **Best `regParam`**: 0.01
- **Best `maxIter`**: 50
- **Best Validation RMSE**: 7.052517952145757

#### Analysis

- The lowest RMSE was found with
