import copy

def return_best_online_model(starting_model, df, train_size=0.6,  grace_periods = [50, 100, 500, 1000, 5000, 10000, 25000, 50000], task = "regression"):
    
    len_train = int(train_size * len(df))
    df_train = df[:len_train]
    df_val = df[len_train:]

    best_model = None
    if task == "regression":
        best_error = float('inf')
    if task == "classification":
        best_error = 0

    # Iterate over different grace periods
    for gp in grace_periods:
        model_gp = copy.deepcopy(starting_model)
        model_gp.grace_period = gp

        # Train model on training set
        for _, row in df_train.iterrows():
            x = row.drop('target')
            y = row['target']
            model_gp.learn_one(x.to_dict(), y)

        if task == "regression":
            # Validate model on validation set
            total_error = 0
            for _, row in df_val.iterrows():
                x = row.drop('target')
                y = row['target']
                y_pred = model_gp.predict_one(x.to_dict())
                total_error += abs(y - (y_pred if y_pred is not None else 0))
            
            # Select best model
            if total_error < best_error:
                best_error = total_error
                best_model = model_gp
        
        if task == "classification":
            
            correct_predictions = 0
            total_predictions = 0
            
            # Validate model on validation set
            for _, row in df_val.iterrows():
                x = row.drop('target')
                y = row['target']
                y_pred = model_gp.predict_one(x.to_dict())
                
                if y_pred is not None and y_pred == y:
                    correct_predictions += 1
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Select best model
            if accuracy > best_error:  # Higher accuracy is better
                best_error = accuracy
                best_model = model_gp

    return best_model