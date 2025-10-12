
# Generate Final Submission

# After training and ensembling:

# 1. Load trained models
model_log = load_model('phase1_log_transform/target_log')
model_sqrt = load_model('phase1_log_transform/target_sqrt')
model_boxcox = load_model('phase1_log_transform/target_boxcox')

# 2. Generate predictions on test set
test_pred_log = model_log.predict(test['catalog_content'])
test_pred_log = np.expm1(test_pred_log)  # Inverse transform

test_pred_sqrt = model_sqrt.predict(test['catalog_content'])
test_pred_sqrt = test_pred_sqrt ** 2  # Inverse transform

test_pred_boxcox = model_boxcox.predict(test['catalog_content'])
test_pred_boxcox = pt.inverse_transform(test_pred_boxcox)  # Inverse transform

# 3. Ensemble with optimal weights
optimal_weights = (0.5, 0.2, 0.3)  # From validation
final_predictions = ensemble_predictions(
    test_pred_log, 
    test_pred_sqrt, 
    test_pred_boxcox, 
    optimal_weights
)

# 4. Create submission file
submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_predictions
})

# 5. Ensure positive prices
submission['price'] = submission['price'].clip(lower=0.01)

# 6. Save
submission.to_csv('test_out_phase1_log_ensemble.csv', index=False)

print(f'Submission saved: {len(submission)} predictions')
print(f'Price range: ${submission["price"].min():.2f} - ${submission["price"].max():.2f}')
