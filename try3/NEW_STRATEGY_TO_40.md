# 🎯 NEW STRATEGY: Enhance Huber Loss to <40% SMAPE

## 📊 Current Status

**Huber Loss Baseline: 47.378%** ✅ (BEST MODEL!)
- Two-Stage V1: 54.237% ❌
- Two-Stage V2: 52.247% ❌

**Gap to <40%: 7.378 points**

---

## ❌ Why Two-Stage Failed

1. **Classification errors cascade** (25% misclassified)
2. **Hard boundaries** ($19.99 vs $20.01 treated differently)
3. **Lost shared information** (independent regression heads)
4. **Overfitting to ranges** (not generalizable)

**Conclusion:** Two-stage is fundamentally flawed for this problem.

---

## ✅ NEW APPROACH: Multi-Task Learning

Instead of two-stage (classify THEN regress), use **joint training**:

### Architecture:
```
DistilBERT Embeddings (768-dim)
        ↓
    Shared Features
        ↓
    ┌────────┴────────┐
    │                 │
Price Range      Price Value
(soft logits)    (regression)
    │                 │
    └────────┬────────┘
             ↓
    Range-Aware Loss
```

### Key Differences from Two-Stage:

| Feature | Two-Stage ❌ | Multi-Task ✅ |
|---------|-------------|--------------|
| Classification | Hard (argmax) | Soft (probabilities) |
| Regression | Conditional | Unconditional |
| Information flow | Blocked | Shared |
| Boundaries | Hard ($20) | Soft (gradual) |
| Training | Sequential | Joint |

---

## 🚀 Implementation Plan

### Step 1: Multi-Task Huber Loss
```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        self.distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.shared = nn.Linear(768, 256)
        
        # Auxiliary task: price range (helps learning)
        self.range_head = nn.Linear(256, 3)
        
        # Main task: price regression
        self.price_head = nn.Linear(256, 1)
    
    def forward(self, input_ids, attention_mask):
        embeddings = self.distilbert(input_ids, attention_mask).last_hidden_state[:, 0]
        shared = F.relu(self.shared(embeddings))
        
        range_logits = self.range_head(shared)  # Soft probabilities
        price_pred = self.price_head(shared)     # Direct regression
        
        return range_logits, price_pred

# Loss: 0.1 * range_loss + 0.9 * huber_loss
```

**Expected: 47.4% → 44-45% SMAPE** (~3 point improvement)

---

### Step 2: Price-Aware Attention
```python
class PriceAwareAttention(nn.Module):
    """
    Add price-specific attention layer
    Learns which words indicate price (premium, bulk, pack, etc.)
    """
    def __init__(self):
        self.attention = nn.MultiheadAttention(768, 8)
        self.price_query = nn.Parameter(torch.randn(1, 1, 768))
    
    def forward(self, embeddings):
        # Query: "What indicates price?"
        price_context, _ = self.attention(
            self.price_query.expand(embeddings.size(0), -1, -1),
            embeddings,
            embeddings
        )
        return price_context
```

**Expected: 44-45% → 42-43% SMAPE** (~2 point improvement)

---

### Step 3: Quantile Regression
```python
class QuantileHuberLoss(nn.Module):
    """
    Combine Quantile Loss + Huber Loss
    - Quantile: Better for SMAPE (asymmetric)
    - Huber: Robust to outliers
    """
    def __init__(self, quantile=0.5, delta=1.0):
        self.quantile = quantile
        self.delta = delta
    
    def forward(self, pred, target):
        error = target - pred
        
        # Quantile component
        quantile_loss = torch.where(
            error > 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        
        # Huber component (clip large errors)
        huber_error = torch.where(
            torch.abs(error) <= self.delta,
            0.5 * error ** 2,
            self.delta * (torch.abs(error) - 0.5 * self.delta)
        )
        
        return torch.mean(quantile_loss + huber_error)
```

**Expected: 42-43% → 40-41% SMAPE** (~2 point improvement)

---

### Step 4: Ensemble (Final Push)
```python
# Combine 3 models:
# 1. Huber Loss (47.4%)
# 2. Multi-Task (44-45%)
# 3. Quantile-Huber (40-41%)

final_pred = 0.3 * huber + 0.3 * multitask + 0.4 * quantile
```

**Expected: 40-41% → 38-39% SMAPE** (<40% ✅)

---

## 📈 Projected Timeline

### This Week: Multi-Task Model
- **Day 1**: Implement multi-task architecture
- **Day 2**: Train and validate (target: 44-45%)
- **Expected**: ~3 point improvement

### Next Week: Price-Aware Attention
- **Day 1**: Add attention layer
- **Day 2**: Train and validate (target: 42-43%)
- **Expected**: ~2 point improvement

### Week After: Quantile-Huber
- **Day 1**: Implement quantile-huber loss
- **Day 2**: Train and validate (target: 40-41%)
- **Expected**: ~2 point improvement

### Final Week: Ensemble
- **Day 1**: Ensemble optimization
- **Day 2**: Final submission (target: <40%)
- **Expected**: **<40% SMAPE ✅**

---

## 🎯 Quick Decision

### Option 1: Start Multi-Task NOW ⭐ RECOMMENDED
```bash
# I'll create the multi-task model
# Expected: 44-45% SMAPE in ~2-3 hours
# Improvement: ~3 points from 47.4%
```

### Option 2: Try Different Loss Function
```bash
# Try Quantile-Huber directly
# Faster to implement (30 min)
# Expected: 45-46% SMAPE
# Improvement: ~2 points from 47.4%
```

### Option 3: Add Image Features Back
```bash
# Use Huber Loss + Image features
# You have images already extracted
# Expected: 45-46% SMAPE
# Improvement: ~2 points from 47.4%
```

---

## 💡 My Recommendation

**Try Option 2 FIRST (Quantile-Huber Loss) - 30 minutes!**

Why?
1. ✅ Fastest to implement and test
2. ✅ Builds on working Huber baseline
3. ✅ Expected ~2 point improvement → 45-46%
4. ✅ If works, add multi-task for another 3 points → 42-43%
5. ✅ Total: 47.4% → 42-43% in <1 day!

Then:
- Multi-task (Day 2): 42-43% → 40-41%
- **<40% ACHIEVED!** 🎉

---

## 🚀 Want me to create Quantile-Huber model now?

It's only a **10-line change** to your current Huber model!

**Expected time:** 30 min implementation + 2.5 hours training = **3 hours total**
**Expected result:** 45-46% SMAPE (-2 points from 47.4%)

Then we can add multi-task tomorrow for another -3 points → 42-43%!

**Ready to proceed?**
