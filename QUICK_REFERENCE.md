# Code Quality Issues - Quick Reference

**Use this as a checklist and quick lookup guide for all 26 issues identified**

---

## CRITICAL ISSUES (Fix Immediately)

### 🔴 Issue 1: Duplicate `build_lstm_model()` and `build_bi_lstm_model()` 
- **File:** `drowsiness_detection/models.py`
- **Lines:** 163, 204 (original) and 276, 301 (duplicates)
- **Fix:** Delete lines 276-325
- **Status:** Not Started ☐

### 🔴 Issue 2: Undefined Variable `KSS_THRESHOLD`
- **File:** `drowsiness_detection/data.py`
- **Line:** 151
- **Fix:** Create `constants.py`, use `KSS_THRESHOLD_BINARY`
- **Status:** Not Started ☐

### 🔴 Issue 3: No Tests for `data.py` (609 lines)
- **File:** `tests/test_data.py` (doesn't exist)
- **Coverage:** 0%
- **Fix:** Create comprehensive test file
- **Status:** Not Started ☐

### 🔴 Issue 4: Magic Numbers Throughout Codebase
- **Files:** Multiple (data.py, models.py, visualize.py, etc.)
- **Count:** 8+ different magic values
- **Fix:** Move to `constants.py`
- **Status:** Not Started ☐

### 🔴 Issue 5: Global State in `config.py`
- **File:** `drowsiness_detection/config.py`
- **Lines:** 10, 64
- **Fix:** Document, plan refactoring
- **Status:** Not Started ☐

---

## HIGH PRIORITY ISSUES (This Week)

### 🟠 Issue 6: Duplicate Label Discretization Logic
- **File:** `drowsiness_detection/data.py`
- **Lines:** 234-252 (inline) vs 263-282 (function)
- **Fix:** Replace inline with function call
- **Status:** Not Started ☐

### 🟠 Issue 7: Typo - `create_emtpy_array_of_max_size`
- **File:** `drowsiness_detection/helpers.py`
- **Line:** 128
- **Fix:** Rename to `create_empty_array_of_max_size`
- **Status:** Not Started ☐

### 🟠 Issue 8: Unused Import - `deepcopy`
- **File:** `drowsiness_detection/visualize.py`
- **Line:** 1
- **Fix:** Delete the line
- **Status:** Not Started ☐

### 🟠 Issue 9: 20+ Print Statements Instead of Logging
- **Files:** metrics.py, models.py, helpers.py, visualize.py
- **Count:** 20+ occurrences
- **Fix:** Replace with logger calls
- **Status:** Not Started ☐

### 🟠 Issue 10: Missing Tests for `visualize.py` (741 lines)
- **Coverage:** 0%
- **Fix:** Create basic test suite
- **Status:** Not Started ☐

---

## MEDIUM PRIORITY ISSUES (Next Week)

### 🟡 Issue 11: Missing Tests for `config.py` (64 lines)
- **Coverage:** 0%
- **Status:** Not Started ☐

### 🟡 Issue 12: Missing Tests for `helpers.py` (205 lines - mostly untested)
- **Coverage:** ~25% (only 3/10 functions tested)
- **Status:** Not Started ☐

### 🟡 Issue 13: Weak Exception Messages
- **File:** `drowsiness_detection/run_grid_search_experiment.py`
- **Lines:** 318, 328, 340
- **Fix:** Add descriptive error messages
- **Status:** Not Started ☐

### 🟡 Issue 14: Silent Failure - Returns `None` Instead of Raising
- **File:** `drowsiness_detection/data.py`
- **Function:** `get_kss_labels_for_feature_file()`
- **Fix:** Raise `FileNotFoundError` instead
- **Status:** Not Started ☐

### 🟡 Issue 15: Missing Input Validation
- **Functions:** `get_train_test_splits()`, model builders, etc.
- **Count:** 10+ functions need validation
- **Status:** Not Started ☐

### 🟡 Issue 16: Monolithic `data.py` (609 lines)
- **Current:** Single file with mixed responsibilities
- **Suggestion:** Consider splitting into submodules
- **Status:** Nice to Have ☐

### 🟡 Issue 17: Complex Function `train_test_split_by_subjects()`
- **File:** `drowsiness_detection/data.py`
- **Lines:** 412-466 (54 lines)
- **Fix:** Extract helper functions
- **Status:** Not Started ☐

### 🟡 Issue 18: Inconsistent Type Annotations
- **Files:** helpers.py, models.py (duplicates)
- **Fix:** Add complete type hints
- **Status:** Not Started ☐

### 🟡 Issue 19: Return Type Mismatch
- **File:** `drowsiness_detection/data.py`
- **Function:** `get_train_test_splits()`
- **Line:** 130 (type hint says 2 returns, actually returns 4)
- **Status:** Not Started ☐

---

## LOW PRIORITY ISSUES (Polish)

### 🟢 Issue 20: Import Organization Not Following PEP 8
- **File:** `drowsiness_detection/visualize.py`
- **Fix:** Reorder: std lib → third-party → local
- **Status:** Not Started ☐

### 🟢 Issue 21: Inconsistent Learning Rate Defaults
- **Files:** models.py
- **Issue:** 0.001 vs 0.002 without explanation
- **Status:** Not Started ☐

### 🟢 Issue 22: Missing Docstring Examples
- **Functions:** Multiple data loading/preprocessing functions
- **Status:** Not Started ☐

### 🟢 Issue 23: Version Pinning Too Loose
- **File:** `requirements.txt`
- **Issue:** Uses ~= instead of ==
- **Status:** Not Started ☐

### 🟢 Issue 24: Hardcoded Config in `run.sh`
- **File:** `run.sh`
- **Issue:** Magic numbers in commented examples
- **Status:** Not Started ☐

### 🟢 Issue 25: Incomplete Comments
- **File:** `drowsiness_detection/run_grid_search_experiment.py`
- **Line:** 180 (commented code without explanation)
- **Status:** Not Started ☐

### 🟢 Issue 26: Unused Variable
- **File:** `drowsiness_detection/run_grid_search_experiment.py`
- **Line:** 406 (class_weights assigned but never used)
- **Status:** Not Started ☐

---

## Implementation Progress Tracker

### Phase 1: Critical Fixes (2-3 hours)
- [ ] Issue 1: Delete duplicate functions
- [ ] Issue 2: Fix KSS_THRESHOLD
- [ ] Issue 7: Fix typo
- [ ] Issue 8: Remove unused import
- [ ] Issue 4: Create constants.py
- [ ] Issue 6: Remove duplicate logic

**Estimate:** 2-3 hours  
**Status:** Not Started ☐

### Phase 2: Logging & Configuration (3-4 hours)
- [ ] Create logging_config.py
- [ ] Issue 9: Replace print statements with logging
- [ ] Document magic numbers in constants.py

**Estimate:** 3-4 hours  
**Status:** Not Started ☐

### Phase 3: Validation & Error Handling (2-3 hours)
- [ ] Issue 14: Fix silent failures
- [ ] Issue 15: Add input validation
- [ ] Issue 13: Improve exception messages
- [ ] Issue 19: Fix return type annotations

**Estimate:** 2-3 hours  
**Status:** Not Started ☐

### Phase 4: Documentation (1-2 hours)
- [ ] Issue 18: Add type hints
- [ ] Issue 22: Add docstring examples
- [ ] Issue 25: Complete comments

**Estimate:** 1-2 hours  
**Status:** Not Started ☐

### Phase 5: Testing (3-4 hours)
- [ ] Issue 3: Create tests for data.py
- [ ] Issue 10: Create tests for visualize.py
- [ ] Issue 11: Create tests for config.py
- [ ] Issue 12: Complete tests for helpers.py

**Estimate:** 3-4 hours  
**Status:** Not Started ☐

### Polish (Optional - 2-3 hours)
- [ ] Issue 20: Fix import organization
- [ ] Issue 21: Document inconsistencies
- [ ] Issue 23: Tighten version pinning
- [ ] Issue 24: Document run.sh
- [ ] Issue 26: Remove unused variables

**Estimate:** 2-3 hours  
**Status:** Not Started ☐

---

## Quick Fix Checklist (Can Do in 30 Minutes)

These can be done immediately with minimal risk:

- [ ] **5 min** - Delete lines 276-325 in models.py (Issue 1)
- [ ] **2 min** - Remove line 1 in visualize.py (Issue 8)  
- [ ] **5 min** - Rename function in helpers.py (Issue 7)
- [ ] **10 min** - Create constants.py (Issue 4)
- [ ] **5 min** - Fix KSS_THRESHOLD import (Issue 2)

**Total Time: ~27 minutes**

After these 5 tasks, the codebase becomes more consistent and the critical bug is fixed.

---

## By-File Issue Count

| File | Issues | Critical | High | Medium | Low |
|------|--------|----------|------|--------|-----|
| models.py | 1 | 1 | - | - | - |
| data.py | 5 | 1 | 1 | 2 | 1 |
| helpers.py | 2 | - | 1 | - | 1 |
| visualize.py | 3 | - | 1 | 1 | 1 |
| config.py | 1 | 1 | - | - | - |
| metrics.py | 1 | - | 1 | - | - |
| run_grid_search_experiment.py | 5 | - | - | 3 | 2 |
| requirements.txt | 1 | - | - | - | 1 |
| run.sh | 1 | - | - | - | 1 |
| **TOTAL** | **26** | **5** | **5** | **10** | **6** |

---

## Estimated Completion Timeline

**Week 1:**
- Complete Phase 1 (critical fixes) - 2-3 hours
- Start Phase 2 (logging) - 2-3 hours
- **Total: 4-6 hours**

**Week 2:**
- Complete Phase 2 - 1-2 hours
- Complete Phase 3 (validation) - 2-3 hours
- Start Phase 5 (testing) - 2 hours
- **Total: 5-7 hours**

**Week 3:**
- Complete Phase 4 - 1-2 hours
- Complete Phase 5 - 2 hours
- Polish (optional) - 1-2 hours
- **Total: 4-6 hours**

**Grand Total: 13-19 hours** (Median: 16 hours or 2 work days)

---

## How to Use This Document

1. **Print or bookmark** this page for quick reference
2. **Check off items** as you complete them
3. **Follow the "Implementation Guide"** for step-by-step instructions
4. **Reference "Code Quality Analysis"** for detailed explanations
5. **Run validation commands** after each phase

---

## Validation Commands

```bash
# After each phase, run:
python -m pytest tests/ -v

# After Phase 2, verify no print statements:
grep -r "print(" drowsiness_detection/ --include="*.py" | grep -v test_

# After Phase 5, check coverage:
python -m pytest --cov=drowsiness_detection tests/

# After all phases, run linters:
python -m mypy drowsiness_detection/ --ignore-missing-imports
python -m flake8 drowsiness_detection/
```

---

## Support Reference

- **Detailed Issue Descriptions:** CODE_QUALITY_ANALYSIS.md
- **Step-by-Step Implementation:** IMPLEMENTATION_GUIDE.md
- **High-Level Overview:** ANALYSIS_SUMMARY.md

