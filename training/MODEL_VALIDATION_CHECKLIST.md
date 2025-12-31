# Model Validation Checklist

Before deploying any new model to production, validate it against these criteria to prevent issues like excessive false positives.

## Pre-Deployment Validation

### 1. Check Training Distribution vs Production

**Problem**: Training with aggressive undersampling (4:1 green:yellow) doesn't match production (19:1 ratio).

**Solution**: Validate that training distribution is reasonable:

```bash
# In training_history.json, check class_distribution
# Green:Yellow ratio should be 6:1 to 10:1 (not 4:1)
```

**Red flags**:
- ❌ Green:Yellow ratio < 5:1 (too aggressive undersampling)
- ❌ Total samples dropped by >40% (excessive undersampling)

### 2. Review Test Set Metrics

**Required metrics** (from training_history.json):

| Metric | Minimum Acceptable | Ideal |
|--------|-------------------|-------|
| Yellow Precision | >85% | >90% |
| Yellow Recall | >60% | >70% |
| Overall Accuracy | >94% | >96% |

**Red flags**:
- ❌ Yellow precision < 85% (too many false positives)
- ❌ Green→Yellow error rate > 1% (false alarm problem)

### 3. Production Simulation Test

Before deploying, simulate production with a production-realistic validation set:

```python
# Add to train_model.py - production validation
# Use ORIGINAL unbalanced distribution (not undersampled)
prod_validation_data = load_training_data()  # Don't undersample!

# Test model on realistic distribution
prod_results = evaluate_on_production_distribution(model, prod_validation_data)

# Require:
# - False positive rate < 2% on greens
# - Yellow precision > 80% in production distribution
```

### 4. Manual Spot Check

Before deploying:

```bash
# Test on 10-20 random recent images
python -m api.check_status_job --test-mode

# Manually verify predictions match reality
# Look for false yellows on obviously green images
```

## Training Configuration Guidelines

### Undersampling Ratio (train_model.py:469)

**Current dangerous value**:
```python
TARGET_RATIO = 4.0  # TOO AGGRESSIVE - caused Dec 30 false positives
```

**Safe values**:
```python
TARGET_RATIO = 8.0   # Balanced (recommended for production)
TARGET_RATIO = 10.0  # Conservative (very safe, may miss edge cases)
TARGET_RATIO = 6.0   # Aggressive but safer than 4.0
```

**Rule**: Never go below 6:1 unless you have >500 yellow samples.

### Yellow Class Weight Boost (train_model.py:510)

**Current**:
```python
YELLOW_BOOST_FACTOR = 1.5  # Additional 50% boost
```

**Safe range**: 1.3 - 1.8
- Lower (1.3) = Fewer false positives, may miss some yellows
- Higher (1.8) = Catch more yellows, risk false positives

### Decision Threshold Default (lib/muni_lib.py:366)

**Safe production default**:
```python
YELLOW_THRESHOLD = 0.75  # Start here, not 0.70
```

Can lower to 0.70 if missing too many real yellows.

## Deployment Process

### Step 1: Review Metrics
```bash
# Check latest training run
cat artifacts/models/training_history.json | tail -100

# Verify:
# - Yellow precision > 85%
# - Green→Yellow errors < 1%
# - Reasonable class distribution
```

### Step 2: Snapshot Current Production Model
```bash
# Before deploying new model, snapshot current production model
cp -r artifacts/models/v1 artifacts/models/v1_prod_backup_$(date +%Y%m%d)
```

### Step 3: Test Locally First
```bash
# Copy new model to v1/
cp -r artifacts/models/snapshots/YYYYMMDD_HHMMSS/model artifacts/models/v1

# Test locally
python -m api.check_status_job

# Verify output makes sense
```

### Step 4: Deploy with Monitoring
```bash
./deploy/cloud/deploy-services.sh

# Monitor for 24-48 hours
# Watch for false yellow spike in logs
```

### Step 5: Rollback Procedure

If issues detected:
```bash
# Restore previous production model
mv artifacts/models/v1 artifacts/models/v1_failed_$(date +%Y%m%d)
cp -r artifacts/models/v1_prod_backup_YYYYMMDD artifacts/models/v1

# Redeploy
./deploy/cloud/deploy-services.sh
```

## Common Issues and Solutions

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Too many false yellows | Aggressive undersampling | Increase TARGET_RATIO to 8.0+ |
| Missing real yellows | Conservative training | Add more yellow samples, lower YELLOW_THRESHOLD |
| High confidence errors | Mislabeled training data | Review and fix labels, retrain |
| Inconsistent predictions | Class imbalance | Adjust class weights, not just undersampling |

## Model Performance History

Track these metrics over time:

| Date | Model | Yellow Precision | Yellow Recall | Green→Yellow FP | Production Status |
|------|-------|-----------------|---------------|----------------|-------------------|
| Dec 23 | v1 | 81.1% | 55.6% | ~0% | Baseline |
| Dec 25 | v2 | **87.2%** | 63.0% | 0.2% | ✅ Production |
| Dec 27 | v3 | 88.9% | 59.3% | 0.7% | Not deployed |
| Dec 29 | v4 | 76.1% | 87.9% | 5.3% | Not deployed (too aggressive) |
| Dec 30 | v5 | 81.7% | 95.1% | 4.2% | ❌ Rolled back (false positives) |

**Lesson**: Dec 25 model (87% precision, 63% recall) was the sweet spot. Dec 30 pushed recall too high at expense of precision.

## Quick Decision Tree

**Should I deploy this model?**

1. Is yellow precision > 85%? 
   - No → Don't deploy, retrain with less aggressive undersampling
   - Yes → Continue

2. Is green→yellow false positive rate < 1%?
   - No → Don't deploy, increase YELLOW_THRESHOLD or retrain
   - Yes → Continue

3. Did you test on 10+ recent images manually?
   - No → Do manual testing first
   - Yes → Safe to deploy with monitoring

4. Do you have a rollback plan?
   - No → Snapshot current model first
   - Yes → Deploy!

## Monitoring After Deployment

**Week 1**: Check daily for false positives
**Week 2-4**: Check 2-3x per week
**Ongoing**: Review monthly, retrain quarterly

**Red flags**:
- Multiple user reports of "says yellow but looks green"
- Yellow alerts happening >10% of the time (should be ~5%)
- Sudden spike in yellow predictions

## Next Training Run Recommendations

Based on Dec 30 failure:

1. **Use TARGET_RATIO = 8.0** (not 4.0)
2. **Start with YELLOW_THRESHOLD = 0.75** (not 0.70)
3. **Target yellow precision > 87%** (match or beat Dec 25)
4. **Accept 60-70% recall** (don't push for 95%+)
5. **Validate on production-like distribution before deploying**

Remember: In production, **precision matters more than recall** for yellow status. Users tolerate missing 1-2 out of 10 yellows, but hate constant false alarms.
