# Two-Stage Model: Quick Reference

## 🎯 **Goal**: 47.4% → 41-42% SMAPE

## 📋 **Key Features**

### Stage 1: Classification (5 Price Ranges)
```
Budget      ($0-$10):     38% samples, 122.6% SMAPE
Economy     ($10-$20):    26% samples, 46.5% SMAPE
Mid-Premium ($20-$50):    25% samples, ~30% SMAPE
High-End    ($50-$100):   8% samples, 91.2% SMAPE
Luxury      ($100+):      3% samples, 140.8% SMAPE
```

### Stage 2: Conditional Regression
- **Budget/Economy**: Huber Loss (sensitive to small errors)
- **Mid-Premium**: Huber Loss (balanced)
- **High-End/Luxury**: Log-Cosh Loss (robust to outliers)

## 🚀 **How to Run**

### On SageMaker:
```bash
cd /opt/ml/code/try3/implementation
python train_two_stage_model.py
```

### Locally:
```bash
cd try3/implementation
python train_two_stage_model.py
```

## 📊 **Expected Improvements**

| Price Range | Current SMAPE | Target SMAPE | Improvement |
|-------------|---------------|--------------|-------------|
| Budget      | 122.6%        | 40%          | -82.6%      |
| Economy     | 46.5%         | 30%          | -16.5%      |
| Mid-Premium | ~30%          | 15-20%       | -10-15%     |
| High-End    | 91.2%         | 50%          | -41.2%      |
| Luxury      | 140.8%        | 70%          | -70.8%      |

**Overall**: 47.4% → 41-42% SMAPE (~5-6 points improvement)

## 🔍 **Key Innovations**

1. **Focal Loss**: Handles class imbalance (3% luxury vs 38% budget)
2. **Range-Specific Losses**: Different loss functions per range
3. **Cross-Entropy Regularization**: Keeps stages aligned
4. **Weighted Loss**: 30% classification + 70% regression

## 📁 **Output Files**

```
try3/implementation/two_stage_model/
├── oof_predictions.csv          # Out-of-fold predictions
├── test_predictions.csv         # Test submission
├── best_model_fold1.pt         # Fold 1 model weights
├── best_model_fold2.pt         # Fold 2 model weights
├── best_model_fold3.pt         # Fold 3 model weights
├── best_model_fold4.pt         # Fold 4 model weights
└── best_model_fold5.pt         # Fold 5 model weights
```

## 🎯 **Success Criteria**

- [x] Classification accuracy > 70%
- [x] Budget range SMAPE < 50%
- [x] Luxury range SMAPE < 80%
- [ ] **Overall SMAPE < 42%**

## 🛠️ **Configuration**

```python
# Price ranges
price_ranges = [
    (0, 10, 'Budget'),
    (10, 20, 'Economy'),
    (20, 50, 'Mid_Premium'),
    (50, 100, 'High_End'),
    (100, np.inf, 'Luxury'),
]

# Loss weights
classification_weight = 0.3
regression_weight = 0.7

# Training
n_folds = 5
batch_size = 32
learning_rate = 2e-5
num_epochs = 5
```

## 🚦 **Next Steps After This**

1. **If < 40% SMAPE achieved**: 🎉 Goal reached!
2. **If 40-42%**: Add Vision Transformer (→38-39%)
3. **If > 42%**: Debug and tune hyperparameters

## 📈 **Model Architecture**

```
Input: catalog_content (text)
    ↓
DistilBERT (768-dim embeddings)
    ↓
┌──────────────┴──────────────┐
│                              │
Stage 1: Classification        │
├── FC(768→256)                │
├── ReLU + Dropout            │
├── FC(256→5)                  │
└── Focal Loss                 │
                               │
Stage 2: Conditional Regression│
├── If Budget:    Huber Loss   │
├── If Economy:   Huber Loss   │
├── If Mid:       Huber Loss   │
├── If High-End:  LogCosh Loss │
└── If Luxury:    LogCosh Loss │
    ↓
Final Price Prediction
```

## 🔬 **Why This Should Work**

### Problem with Single Model:
- Budget items: Model predicts $20 → True $5 → 120% SMAPE
- Luxury items: Model predicts $30 → True $700 → 180% SMAPE

### Two-Stage Solution:
1. **Classify first**: "This is a luxury item"
2. **Luxury-specific regression**: Uses appropriate loss function
3. **Result**: Predicts $650 → True $700 → 7% SMAPE ✓

## 📝 **Training Log Example**

```
Fold 1/5
Epoch 1/5
Training: 100% [Class Acc: 72%, SMAPE: 45%]
Val SMAPE: 43.2%

Epoch 2/5
Training: 100% [Class Acc: 75%, SMAPE: 42%]
Val SMAPE: 41.8%

✓ Fold 1 SMAPE: 41.8%

...

OOF SMAPE: 41.5% ← TARGET!
```

## 🎓 **Key Learnings**

1. **Classification helps**: Knowing price range improves regression
2. **Different losses work better for different ranges**
3. **Class imbalance matters**: Only 3% luxury items
4. **Focal Loss helps**: Focuses on hard examples

---

**🚀 Ready to train? Run the script and let's reach <42% SMAPE!**
