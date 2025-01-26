`residuals_heter/` contains the parameters for the dyadic environment simulator models fitted using generalized estimating equation (GEE).

## Data Cleaning Procedure

### Generate daily variables

-   The *daily heart rate* is the average of `morningHEART`, `dayHEART` and `nightHEART`. So is the *daily mood*.
-   The *daily sleep* is the sum of `morningSLEEP`, `nightSLEEP`.
-   The *daily step count* is the sum of `daySTEPS`, `nightSTEPS`. We further take a squre root on the *daily step count*.

### Valid dyads

We only include dyads that meet certain threshold. Here is the list of rules we use to filter dyads:

-   Total number of rows larger than $7 \times 15$ (15 weeks of interactions)
-   The proportion of missing for each variable in the first 15 weeks of both participant in this dyads has to be smaller than $0.75$

This results in 49 valid dyads

### Data imputation

We use `complete` function in `tidyr` and `mice` function in `mice` to impute missing data.

### Creating weekly variables

We calculate the mean of `dailymood`, whose shape is $7 \times 15$, resulting a `weeklymood` of shape $15$. This creates the variable `weekly_mood_patient_next`. The variable `weeklymood` is created by discarding the last weekly mood and appending an `NA` to the front.

To make sense of the dataset, the first week is discarded and the resulting dataset has 14 weeks.

### Standardize

All the variables are standardized by each column with mean zero and standard deviation one for the full dataset that pools across dyads.
