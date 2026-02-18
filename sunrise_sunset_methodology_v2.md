# Sunrise-Sunset Skill Classification Methodology

## Technical Documentation for Implementation

---

## 1. Overview

**Purpose:** Classify skills as Sunrise (growing importance, scarce supply) or Sunset (declining importance, oversupplied) using a percentile-based relative scoring system that adapts dynamically to data distribution.

**Output:** Each skill receives a composite score (0-100) and categorical classification.

**Key Benefits:**
- No hard-coded thresholds that break with noisy data
- Adapts to industry-specific baselines automatically
- Handles variation in job roles and data quality gracefully
- Reconciles mismatched skill lists across multiple data sources

---

## 2. Input Data Schema

### 2.1 Three Separate Input Tables

The methodology accepts three separate input files that may contain different skill sets:

**Table A: Supply**
| Field | Data Type | Required | Description |
|-------|-----------|----------|-------------|
| skill_name | string | Yes | Skill identifier |
| sub_family | string | Optional | Grouping category for skills |
| talent_pool_size | integer | Yes | Current employed talent count with this skill |

**Table B: Demand Current Year**
| Field | Data Type | Required | Description |
|-------|-----------|----------|-------------|
| skill_name | string | Yes | Skill identifier |
| job_openings_current | integer | Yes | Job postings or demand signals from current year |

**Table C: Demand Previous Year**
| Field | Data Type | Required | Description |
|-------|-----------|----------|-------------|
| skill_name | string | Yes | Skill identifier |
| job_openings_previous | integer | Yes | Job postings or demand signals from previous year |

### 2.2 Sample Input Formats

**Supply Table (CSV):**

```csv
skill_name,sub_family,talent_pool_size
Microarchitecture,Silicon & SoC Design,2400
ASIC Design,Silicon & SoC Design,1800
Firmware Development,Embedded Software & Firmware,12000
Test Planning,Test Quality & Compliance,8500
```

**Demand Current Year Table (CSV):**

```csv
skill_name,job_openings_current
Microarchitecture,4600
ASIC Design,2350
Firmware Development,10200
Test Planning,4100
```

**Demand Previous Year Table (CSV):**

```csv
skill_name,job_openings_previous
Microarchitecture,3800
ASIC Design,2200
Firmware Development,9500
Test Planning,4200
```

---

## 3. Step 0: Data Reconciliation

### 3.1 Problem Statement

Three separate input tables may contain different skill sets:
- Supply Table: ~500 skills with talent pool size
- Demand Current Year Table: ~500 skills with current year job openings
- Demand Previous Year Table: ~500 skills with previous year job openings

Skills may not match 1:1 across tables due to naming variations, new/deprecated skills, or data gaps.

### 3.2 Skill Name Normalization

Before matching, normalize all skill names to handle minor variations:

```python
import re

def normalize_skill_name(skill_name):
    """
    Normalize skill names for matching across tables.
    Handles case differences, extra spaces, and special characters.
    """
    if skill_name is None:
        return None
    
    normalized = skill_name.lower().strip()
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)  # Remove special characters
    normalized = re.sub(r'\s+', ' ', normalized)  # Collapse multiple spaces
    return normalized
```

**Examples:**

| Original | Normalized |
|----------|------------|
| "Microarchitecture" | "microarchitecture" |
| "ASIC Design" | "asic design" |
| "C++" | "c" |
| "IoT (Internet of Things)" | "iot internet of things" |
| "  Firmware  Development " | "firmware development" |

### 3.3 Skill Matching Logic

**Step 3.3.1: Create Normalized Keys**

```python
def add_normalized_column(df, skill_column='skill_name'):
    """
    Add normalized skill name column to dataframe.
    """
    df['skill_name_normalized'] = df[skill_column].apply(normalize_skill_name)
    return df

# Apply to all three tables
supply_table = add_normalized_column(supply_table)
demand_current_table = add_normalized_column(demand_current_table)
demand_previous_table = add_normalized_column(demand_previous_table)
```

**Step 3.3.2: Find Common Skills (Inner Join)**

```python
def find_common_skills(supply_df, demand_current_df, demand_previous_df):
    """
    Find skills present in all three tables.
    Returns set of normalized skill names.
    """
    skills_supply = set(supply_df['skill_name_normalized'].dropna())
    skills_demand_current = set(demand_current_df['skill_name_normalized'].dropna())
    skills_demand_previous = set(demand_previous_df['skill_name_normalized'].dropna())
    
    common_skills = skills_supply & skills_demand_current & skills_demand_previous
    
    return common_skills, skills_supply, skills_demand_current, skills_demand_previous
```

**Step 3.3.3: Create Merged Analysis Table**

```python
def create_merged_table(supply_df, demand_current_df, demand_previous_df, common_skills):
    """
    Merge three tables on common skills.
    Preserves original skill name from supply table.
    """
    # Filter to common skills
    supply_filtered = supply_df[supply_df['skill_name_normalized'].isin(common_skills)]
    demand_current_filtered = demand_current_df[demand_current_df['skill_name_normalized'].isin(common_skills)]
    demand_previous_filtered = demand_previous_df[demand_previous_df['skill_name_normalized'].isin(common_skills)]
    
    # Merge tables
    merged = supply_filtered.merge(
        demand_current_filtered[['skill_name_normalized', 'job_openings_current']],
        on='skill_name_normalized',
        how='inner'
    )
    
    merged = merged.merge(
        demand_previous_filtered[['skill_name_normalized', 'job_openings_previous']],
        on='skill_name_normalized',
        how='inner'
    )
    
    # Rename columns to standard schema
    merged = merged.rename(columns={
        'job_openings_current': 'demand_current_year',
        'job_openings_previous': 'demand_previous_year'
    })
    
    # Select final columns
    final_columns = ['skill_name', 'sub_family', 'talent_pool_size', 
                     'demand_current_year', 'demand_previous_year']
    
    # Handle missing sub_family column
    if 'sub_family' not in merged.columns:
        merged['sub_family'] = None
    
    return merged[final_columns]
```

### 3.4 Reconciliation Report

Generate a reconciliation summary before proceeding:

```python
def generate_reconciliation_report(skills_supply, skills_demand_current, 
                                    skills_demand_previous, common_skills):
    """
    Generate detailed reconciliation report.
    """
    supply_only = skills_supply - skills_demand_current - skills_demand_previous
    demand_current_only = skills_demand_current - skills_supply - skills_demand_previous
    demand_previous_only = skills_demand_previous - skills_supply - skills_demand_current
    
    # Skills in exactly 2 tables
    supply_and_current_only = (skills_supply & skills_demand_current) - skills_demand_previous
    supply_and_previous_only = (skills_supply & skills_demand_previous) - skills_demand_current
    current_and_previous_only = (skills_demand_current & skills_demand_previous) - skills_supply
    
    total_unique = len(skills_supply | skills_demand_current | skills_demand_previous)
    
    report = {
        "reconciliation_summary": {
            "supply_total": len(skills_supply),
            "demand_current_total": len(skills_demand_current),
            "demand_previous_total": len(skills_demand_previous),
            "common_skills_count": len(common_skills),
            "total_unique_skills": total_unique,
            "match_rate_pct": round(len(common_skills) / total_unique * 100, 1) if total_unique > 0 else 0
        },
        "unmatched_counts": {
            "supply_only": len(supply_only),
            "demand_current_only": len(demand_current_only),
            "demand_previous_only": len(demand_previous_only),
            "supply_and_current_only": len(supply_and_current_only),
            "supply_and_previous_only": len(supply_and_previous_only),
            "current_and_previous_only": len(current_and_previous_only)
        },
        "unmatched_skills": {
            "supply_only": sorted(list(supply_only)),
            "demand_current_only": sorted(list(demand_current_only)),
            "demand_previous_only": sorted(list(demand_previous_only)),
            "supply_and_current_only": sorted(list(supply_and_current_only)),
            "supply_and_previous_only": sorted(list(supply_and_previous_only)),
            "current_and_previous_only": sorted(list(current_and_previous_only))
        }
    }
    
    return report
```

**Sample Reconciliation Report Output:**

```json
{
  "reconciliation_summary": {
    "supply_total": 500,
    "demand_current_total": 500,
    "demand_previous_total": 500,
    "common_skills_count": 412,
    "total_unique_skills": 548,
    "match_rate_pct": 75.2
  },
  "unmatched_counts": {
    "supply_only": 23,
    "demand_current_only": 31,
    "demand_previous_only": 18,
    "supply_and_current_only": 42,
    "supply_and_previous_only": 15,
    "current_and_previous_only": 7
  },
  "unmatched_skills": {
    "supply_only": ["skill a", "skill b"],
    "demand_current_only": ["skill c", "skill d"],
    "demand_previous_only": ["skill e", "skill f"],
    "supply_and_current_only": ["skill g", "skill h"],
    "supply_and_previous_only": ["skill i", "skill j"],
    "current_and_previous_only": ["skill k", "skill l"]
  }
}
```

### 3.5 Handling Partial Matches (Optional)

For skills present in 2 of 3 tables, apply special handling:

| Scenario | Tables Present | Classification | Confidence | Handling |
|----------|----------------|----------------|------------|----------|
| New Skill | Supply + Current Demand | Flag as "New Skill" | Low | Cannot calculate growth rate; exclude or use proxy |
| Declining Skill | Supply + Previous Demand | Flag as "Potentially Sunset" | Low | Current demand = 0; flag as Sunset |
| Emerging Skill | Current + Previous Demand | Flag as "Emerging - No Talent Pool" | Low | No supply data; exclude or use proxy |
| Orphan Skill | Only one table | Exclude | N/A | Insufficient data for analysis |

**Implementation:**

```python
def handle_partial_matches(supply_df, demand_current_df, demand_previous_df, 
                           skills_supply, skills_demand_current, skills_demand_previous,
                           include_partial=False):
    """
    Handle skills present in only 2 of 3 tables.
    Returns dataframe with partial matches and appropriate flags.
    """
    if not include_partial:
        return None
    
    partial_matches = []
    
    # Supply + Current only (New Skills)
    supply_and_current = (skills_supply & skills_demand_current) - skills_demand_previous
    for skill in supply_and_current:
        supply_row = supply_df[supply_df['skill_name_normalized'] == skill].iloc[0]
        demand_row = demand_current_df[demand_current_df['skill_name_normalized'] == skill].iloc[0]
        partial_matches.append({
            'skill_name': supply_row['skill_name'],
            'sub_family': supply_row.get('sub_family'),
            'talent_pool_size': supply_row['talent_pool_size'],
            'demand_current_year': demand_row['job_openings_current'],
            'demand_previous_year': 0,
            'partial_match_flag': 'New Skill',
            'confidence': 'Low'
        })
    
    # Supply + Previous only (Declining Skills)
    supply_and_previous = (skills_supply & skills_demand_previous) - skills_demand_current
    for skill in supply_and_previous:
        supply_row = supply_df[supply_df['skill_name_normalized'] == skill].iloc[0]
        demand_row = demand_previous_df[demand_previous_df['skill_name_normalized'] == skill].iloc[0]
        partial_matches.append({
            'skill_name': supply_row['skill_name'],
            'sub_family': supply_row.get('sub_family'),
            'talent_pool_size': supply_row['talent_pool_size'],
            'demand_current_year': 0,
            'demand_previous_year': demand_row['job_openings_previous'],
            'partial_match_flag': 'Potentially Sunset',
            'confidence': 'Low'
        })
    
    # Current + Previous only (Emerging - No Talent Pool)
    current_and_previous = (skills_demand_current & skills_demand_previous) - skills_supply
    for skill in current_and_previous:
        current_row = demand_current_df[demand_current_df['skill_name_normalized'] == skill].iloc[0]
        previous_row = demand_previous_df[demand_previous_df['skill_name_normalized'] == skill].iloc[0]
        partial_matches.append({
            'skill_name': current_row['skill_name'],
            'sub_family': None,
            'talent_pool_size': 0,
            'demand_current_year': current_row['job_openings_current'],
            'demand_previous_year': previous_row['job_openings_previous'],
            'partial_match_flag': 'Emerging - No Talent Pool',
            'confidence': 'Low'
        })
    
    return pd.DataFrame(partial_matches)
```

### 3.6 Process Flow with Reconciliation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                      │
├───────────────────┬───────────────────┬─────────────────────────────────┤
│   Supply Table    │  Demand Current   │      Demand Previous            │
│   (500 skills)    │   (500 skills)    │       (500 skills)              │
└─────────┬─────────┴─────────┬─────────┴───────────────┬─────────────────┘
          │                   │                         │
          ▼                   ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 0.1: NORMALIZE SKILL NAMES                      │
│            • Lowercase, trim, remove special characters                 │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 0.2: FIND COMMON SKILLS                         │
│                  • Inner join on normalized skill names                 │
│                  • Identify unmatched skills per table                  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 STEP 0.3: GENERATE RECONCILIATION REPORT                │
│                  • Match rate, unmatched counts                         │
│                  • List of unmatched skills by category                 │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 0.4: CREATE MERGED TABLE                        │
│               • Combine matched skills into single table                │
│               • Preserve original skill names from supply               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MERGED TABLE                                    │
│              (Only common skills across all 3 tables)                   │
│                        (~400-450 skills)                                │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEPS 1-6: CORE ANALYSIS                             │
│                  • Calculate metrics                                    │
│                  • Convert to percentiles                               │
│                  • Calculate sunrise score                              │
│                  • Classify skills                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Metrics Calculation

### Metric 1: Demand Growth Rate

```python
def calculate_demand_growth_rate(demand_current_year, demand_previous_year):
    """
    Calculate year-over-year demand growth rate.
    Returns decimal (e.g., 0.21 for 21% growth).
    """
    if demand_previous_year is None or demand_previous_year == 0:
        return None
    
    return (demand_current_year - demand_previous_year) / demand_previous_year
```

**Specifications:**
- Output as decimal (e.g., 0.21 for 21% growth)
- Can be negative for declining demand
- Handle edge case: if demand_previous_year = 0, return None

### Metric 2: Supply Pressure Index

```python
def calculate_supply_pressure_index(talent_pool_size, demand_current_year):
    """
    Calculate supply pressure index.
    Lower value = scarcer talent relative to demand.
    """
    if demand_current_year is None or demand_current_year == 0:
        return None
    
    return talent_pool_size / demand_current_year
```

**Specifications:**
- Output as decimal ratio
- Lower value = scarcer talent relative to demand
- Higher value = oversupplied talent relative to demand
- Handle edge case: if demand_current_year = 0, return None

---

## 5. Percentile Conversion

### Step 5.1: Rank Demand Growth Rate

1. Sort all skills by demand_growth_rate in descending order
2. Assign rank (1 = highest growth)
3. Calculate percentile:

```python
def calculate_percentile_rank(values, ascending=False):
    """
    Calculate percentile ranks for a list of values.
    Uses average rank method for ties.
    
    Parameters:
    - values: list of numeric values
    - ascending: if True, lowest value gets highest percentile
    """
    import pandas as pd
    
    series = pd.Series(values)
    
    if ascending:
        ranks = series.rank(method='average', ascending=True)
    else:
        ranks = series.rank(method='average', ascending=False)
    
    total = len(values)
    percentiles = ((total - ranks + 1) / total) * 100
    
    return percentiles.tolist()
```

### Step 5.2: Rank Supply Pressure Index (Inverted)

```python
# For supply pressure, lower value = scarcer = better for sunrise
# So we rank ascending (lowest gets rank 1 = highest percentile)

supply_pressure_percentile = calculate_percentile_rank(
    supply_pressure_values, 
    ascending=True
)
```

**Note:** Inversion ensures that low supply pressure (talent scarcity) results in high percentile, aligning directionally with sunrise classification.

### Handling Tied Ranks

Uses average rank method automatically:

```python
# Example: Values [10, 20, 20, 30]
# Ranks: [4, 2.5, 2.5, 1] (two skills tie at rank 2.5)
```

---

## 6. Composite Sunrise Score Calculation

```python
def calculate_sunrise_score(demand_growth_percentile, supply_pressure_percentile,
                            weight_demand=0.5, weight_supply=0.5):
    """
    Calculate composite sunrise score.
    
    Parameters:
    - demand_growth_percentile: 0-100 score
    - supply_pressure_percentile: 0-100 score (inverted)
    - weight_demand: weight for demand component (default 0.5)
    - weight_supply: weight for supply component (default 0.5)
    
    Returns: sunrise_score (0-100)
    """
    # Validate weights
    if abs((weight_demand + weight_supply) - 1.0) > 0.001:
        raise ValueError("Weights must sum to 1.0")
    
    sunrise_score = (demand_growth_percentile * weight_demand) + \
                    (supply_pressure_percentile * weight_supply)
    
    return round(sunrise_score, 1)
```

### Default Weights

| Parameter | Default Value | Configurable |
|-----------|---------------|--------------|
| weight_demand | 0.5 | Yes |
| weight_supply | 0.5 | Yes |

**Constraint:** weight_demand + weight_supply must equal 1.0

### Preset Weight Configurations

| Preset Name | weight_demand | weight_supply | Use Case |
|-------------|---------------|---------------|----------|
| balanced | 0.5 | 0.5 | Default, general purpose |
| demand_led | 0.6 | 0.4 | Fast-moving markets, growth focus |
| supply_constrained | 0.4 | 0.6 | Tight labor markets, hiring focus |

### Weight Configuration Input Format

```json
{
  "weights": {
    "preset": "balanced"
  }
}
```

Or custom:

```json
{
  "weights": {
    "custom": {
      "demand": 0.55,
      "supply": 0.45
    }
  }
}
```

---

## 7. Classification Logic

### Score-Based Classification

| Min Score | Max Score | Classification | Label Code |
|-----------|-----------|----------------|------------|
| 75 | 100 | Sunrise | SR |
| 55 | 74.99 | Likely Sunrise | LS |
| 45 | 54.99 | Neutral/Transitional | NT |
| 25 | 44.99 | Likely Sunset | LSS |
| 0 | 24.99 | Sunset | SS |

### Implementation Logic

```python
def classify_skill(sunrise_score, thresholds=None):
    """
    Classify skill based on sunrise score.
    
    Parameters:
    - sunrise_score: composite score (0-100)
    - thresholds: optional custom thresholds dict
    
    Returns: (classification, label_code)
    """
    # Default thresholds
    if thresholds is None:
        thresholds = {
            'sunrise': 75,
            'likely_sunrise': 55,
            'neutral': 45,
            'likely_sunset': 25
        }
    
    if sunrise_score >= thresholds['sunrise']:
        return "Sunrise", "SR"
    elif sunrise_score >= thresholds['likely_sunrise']:
        return "Likely Sunrise", "LS"
    elif sunrise_score >= thresholds['neutral']:
        return "Neutral/Transitional", "NT"
    elif sunrise_score >= thresholds['likely_sunset']:
        return "Likely Sunset", "LSS"
    else:
        return "Sunset", "SS"
```

### Custom Threshold Configuration

Allow user to override default classification thresholds:

```json
{
  "thresholds": {
    "sunrise": 75,
    "likely_sunrise": 55,
    "neutral": 45,
    "likely_sunset": 25
  }
}
```

---

## 8. Data Quality and Confidence Flags

### Confidence Flag Assignment

| Condition | Confidence Flag |
|-----------|-----------------|
| Both metrics calculated successfully | High |
| One metric is null or estimated | Medium |
| Demand previous year < 10 (low base) | Medium |
| Either demand value is zero | Low |
| Talent pool size is zero | Low |
| Partial match (from 2 of 3 tables) | Low |

### Implementation Logic

```python
def assign_confidence(demand_growth_rate, supply_pressure_index, 
                      demand_previous_year, demand_current_year, 
                      talent_pool_size, partial_match_flag=None):
    """
    Assign confidence flag based on data quality.
    """
    # Partial matches always get Low confidence
    if partial_match_flag is not None:
        return "Low"
    
    # Check for null or zero values
    if demand_current_year == 0 or demand_current_year is None:
        return "Low"
    
    if talent_pool_size == 0 or talent_pool_size is None:
        return "Low"
    
    if demand_previous_year == 0 or demand_previous_year is None:
        return "Low"
    
    if demand_growth_rate is None or supply_pressure_index is None:
        return "Low"
    
    # Check for low base
    if demand_previous_year < 10:
        return "Medium"
    
    return "High"
```

---

## 9. Edge Case Handling

| Scenario | Handling | Flag |
|----------|----------|------|
| demand_previous_year = 0 | Exclude from ranking or flag as "New Skill" | confidence = Low |
| demand_current_year = 0 | Flag as "Sunset" with confidence = Low | classification = Sunset |
| talent_pool_size = 0 | Flag as "Emerging" with confidence = Low | classification = Sunrise |
| Tied ranks | Use average rank method for ties | N/A |
| Single skill in dataset | Return score = 50, classification = Neutral | confidence = Low |
| Negative demand values | Treat as data error, exclude or flag | error = True |
| Missing required fields | Skip record, log error | error = True |
| Skill not in all 3 tables | Handle via partial match logic or exclude | confidence = Low |

### Implementation Logic

```python
def handle_edge_cases(skill_data):
    """
    Handle edge cases in skill data.
    Returns (flags, errors) tuple.
    """
    errors = []
    flags = []
    
    # Check for negative values
    if skill_data.get('demand_previous_year', 0) < 0 or \
       skill_data.get('demand_current_year', 0) < 0:
        errors.append("Negative demand values detected")
        return flags, errors
    
    if skill_data.get('talent_pool_size', 0) < 0:
        errors.append("Negative talent pool size detected")
        return flags, errors
    
    # Check for zero values
    if skill_data.get('demand_previous_year', 0) == 0:
        flags.append("New Skill - no previous demand data")
    
    if skill_data.get('demand_current_year', 0) == 0:
        flags.append("No current demand - potential sunset")
    
    if skill_data.get('talent_pool_size', 0) == 0:
        flags.append("No existing talent pool - emerging skill")
    
    return flags, errors
```

---

## 10. Output Data Schema

### 10.1 Main Results Table

| Field | Data Type | Description |
|-------|-----------|-------------|
| skill_name | string | Input skill identifier |
| sub_family | string | Input grouping category |
| talent_pool_size | integer | Input value |
| demand_previous_year | integer | Input value |
| demand_current_year | integer | Input value |
| demand_growth_rate | float | Calculated metric (decimal) |
| supply_pressure_index | float | Calculated metric (ratio) |
| demand_growth_percentile | float | Percentile score (0-100) |
| supply_pressure_percentile | float | Percentile score (0-100) |
| sunrise_score | float | Composite score (0-100) |
| classification | string | Category label |
| classification_code | string | Short code (SR, LS, NT, LSS, SS) |
| confidence | string | Data quality flag (High, Medium, Low) |

### 10.2 Sample Output Format (JSON)

```json
{
  "skill_name": "Microarchitecture",
  "sub_family": "Silicon & SoC Design",
  "talent_pool_size": 2400,
  "demand_previous_year": 3800,
  "demand_current_year": 4600,
  "demand_growth_rate": 0.211,
  "supply_pressure_index": 0.52,
  "demand_growth_percentile": 100.0,
  "supply_pressure_percentile": 100.0,
  "sunrise_score": 100.0,
  "classification": "Sunrise",
  "classification_code": "SR",
  "confidence": "High"
}
```

### 10.3 Sample Output Format (CSV)

```csv
skill_name,sub_family,talent_pool_size,demand_previous_year,demand_current_year,demand_growth_rate,supply_pressure_index,demand_growth_percentile,supply_pressure_percentile,sunrise_score,classification,classification_code,confidence
Microarchitecture,Silicon & SoC Design,2400,3800,4600,0.211,0.52,100.0,100.0,100.0,Sunrise,SR,High
```

---

## 11. Optional Features

### 11.1 Sub-Family Level Analysis

Allow percentile calculation within sub-family groupings instead of across entire dataset.

**Parameter:**

```python
scope = "global" | "sub_family"
```

**Behavior:**
- `global`: Percentiles calculated across all skills in dataset
- `sub_family`: Percentiles calculated within each sub-family separately

### 11.2 Sensitivity Analysis

Run classification with multiple weight combinations to test stability.

**Implementation:**

```python
def sensitivity_analysis(skill_data, weight_combinations=None):
    """
    Run classification with multiple weight combinations.
    
    Parameters:
    - skill_data: dict with demand_growth_percentile and supply_pressure_percentile
    - weight_combinations: list of weight dicts
    
    Returns: dict with results and stability flag
    """
    if weight_combinations is None:
        weight_combinations = [
            {"demand": 0.4, "supply": 0.6},
            {"demand": 0.5, "supply": 0.5},
            {"demand": 0.6, "supply": 0.4}
        ]
    
    results = []
    for weights in weight_combinations:
        score = calculate_sunrise_score(
            skill_data['demand_growth_percentile'],
            skill_data['supply_pressure_percentile'],
            weights['demand'],
            weights['supply']
        )
        classification, code = classify_skill(score)
        results.append({
            "weights": weights,
            "score": score,
            "classification": classification
        })
    
    # Check stability
    classifications = [r['classification'] for r in results]
    is_stable = len(set(classifications)) == 1
    
    return {
        "results": results,
        "is_stable": is_stable,
        "stability_flag": "Stable" if is_stable else "Sensitive"
    }
```

### 11.3 Time Series Extension

If multi-year data is available, calculate trend-based metrics.

**Additional Input Fields:**

```json
{
  "demand_year_minus_2": 3200,
  "demand_year_minus_1": 3800,
  "demand_current_year": 4600
}
```

**Additional Metric:**

```python
demand_acceleration = demand_growth_rate_current - demand_growth_rate_previous
```

### 11.4 Export Formats

Support multiple output formats:

| Format | Extension | Use Case |
|--------|-----------|----------|
| CSV | .csv | Spreadsheet analysis |
| JSON | .json | API integration |
| Excel | .xlsx | Business reporting |
| Markdown | .md | Documentation |

---

## 12. Summary Statistics to Generate

| Statistic | Description | Calculation |
|-----------|-------------|-------------|
| total_skills_input | Skills in input tables | count per table |
| common_skills_count | Skills matched across all 3 tables | count(merged_table) |
| match_rate_pct | Percentage of skills matched | common / total_unique * 100 |
| total_skills_analyzed | Count of skills in final analysis | count(results) |
| sunrise_count | Count classified as Sunrise | count(classification = "Sunrise") |
| likely_sunrise_count | Count classified as Likely Sunrise | count(classification = "Likely Sunrise") |
| neutral_count | Count classified as Neutral | count(classification = "Neutral/Transitional") |
| likely_sunset_count | Count classified as Likely Sunset | count(classification = "Likely Sunset") |
| sunset_count | Count classified as Sunset | count(classification = "Sunset") |
| avg_sunrise_score | Mean sunrise score | mean(sunrise_score) |
| median_sunrise_score | Median sunrise score | median(sunrise_score) |
| std_sunrise_score | Standard deviation | std(sunrise_score) |
| high_confidence_pct | Percentage with High confidence | count(confidence = "High") / total * 100 |
| low_confidence_pct | Percentage with Low confidence | count(confidence = "Low") / total * 100 |

### Summary Output Format

```json
{
  "reconciliation": {
    "supply_total": 500,
    "demand_current_total": 500,
    "demand_previous_total": 500,
    "common_skills_count": 412,
    "match_rate_pct": 75.2
  },
  "analysis_summary": {
    "total_skills_analyzed": 412,
    "classification_distribution": {
      "Sunrise": 58,
      "Likely Sunrise": 92,
      "Neutral/Transitional": 84,
      "Likely Sunset": 101,
      "Sunset": 77
    },
    "score_statistics": {
      "mean": 49.8,
      "median": 48.5,
      "std": 26.2,
      "min": 2.4,
      "max": 100.0
    },
    "confidence_distribution": {
      "High": 385,
      "Medium": 19,
      "Low": 8
    },
    "high_confidence_pct": 93.4
  }
}
```

---

## 13. Validation Checks

| Check | Expected Behavior | Error Handling |
|-------|-------------------|----------------|
| All sunrise scores between 0-100 | Pass | Raise ValueError if out of range |
| Classification assigned to every skill | Pass | Raise ValueError if null |
| Percentiles sum logic consistent | Highest rank = highest percentile | Log warning if inconsistent |
| No null values in output for valid inputs | Pass | Flag records with nulls |
| Weights sum to 1.0 | Pass | Raise ValueError if not equal to 1.0 |
| Input files exist and readable | Pass | Raise FileNotFoundError |
| Required columns present in each table | Pass | Raise KeyError with missing columns |
| At least 1 common skill found | Pass | Raise ValueError if no matches |

### Validation Implementation

```python
def validate_inputs(supply_df, demand_current_df, demand_previous_df):
    """
    Validate input dataframes have required columns.
    """
    errors = []
    
    # Check supply table
    if 'skill_name' not in supply_df.columns:
        errors.append("Supply table missing 'skill_name' column")
    if 'talent_pool_size' not in supply_df.columns:
        errors.append("Supply table missing 'talent_pool_size' column")
    
    # Check demand current table
    if 'skill_name' not in demand_current_df.columns:
        errors.append("Demand current table missing 'skill_name' column")
    if 'job_openings_current' not in demand_current_df.columns:
        errors.append("Demand current table missing 'job_openings_current' column")
    
    # Check demand previous table
    if 'skill_name' not in demand_previous_df.columns:
        errors.append("Demand previous table missing 'skill_name' column")
    if 'job_openings_previous' not in demand_previous_df.columns:
        errors.append("Demand previous table missing 'job_openings_previous' column")
    
    if errors:
        raise ValueError(f"Input validation failed: {errors}")
    
    return True


def validate_output(results):
    """
    Validate output results.
    """
    errors = []
    warnings = []
    
    for record in results:
        # Check score range
        if not 0 <= record['sunrise_score'] <= 100:
            errors.append(f"Invalid score for {record['skill_name']}: {record['sunrise_score']}")
        
        # Check classification exists
        if record['classification'] is None:
            errors.append(f"Missing classification for {record['skill_name']}")
        
        # Check percentiles
        if record['demand_growth_percentile'] > 100 or record['supply_pressure_percentile'] > 100:
            errors.append(f"Invalid percentile for {record['skill_name']}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
```

---

## 14. Error Handling and Logging

### Error Codes

| Code | Description | Severity |
|------|-------------|----------|
| E001 | Input file not found | Critical |
| E002 | Required column missing | Critical |
| E003 | Invalid data type in input | Critical |
| E004 | Negative value detected | High |
| E005 | Division by zero attempted | High |
| E006 | Weights do not sum to 1.0 | High |
| E007 | No common skills found across tables | Critical |
| W001 | Low base demand (<10) | Warning |
| W002 | Zero demand detected | Warning |
| W003 | Single skill in dataset | Warning |
| W004 | Tied ranks detected | Info |
| W005 | Low match rate (<50%) | Warning |
| W006 | Partial matches excluded | Info |

### Logging Format

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Example log entries
logging.error("E001 - Input file not found: supply_data.csv")
logging.error("E007 - No common skills found across tables")
logging.warning("W001 - Low base demand for skill: Emerging Tech (demand_previous_year = 5)")
logging.warning("W005 - Low match rate: 42.5% (213 of 500 skills matched)")
logging.info("W004 - Tied ranks detected for skills: Skill A, Skill B at rank 3.5")
```

---

## 15. Command Line Interface Specification

### Basic Usage

```bash
python sunrise_sunset_classifier.py \
  --supply <supply_file> \
  --demand-current <demand_current_file> \
  --demand-previous <demand_previous_file> \
  --output <output_file>
```

### Full Parameter List

| Parameter | Short | Required | Default | Description |
|-----------|-------|----------|---------|-------------|
| --supply | -s | Yes | N/A | Supply table file path (CSV or JSON) |
| --demand-current | -dc | Yes | N/A | Current year demand file path |
| --demand-previous | -dp | Yes | N/A | Previous year demand file path |
| --output | -o | Yes | N/A | Output file path |
| --format | -f | No | csv | Output format (csv, json, xlsx) |
| --weights | -w | No | balanced | Weight preset (balanced, demand_led, supply_constrained) |
| --custom-weights | -cw | No | N/A | Custom weights as "demand:supply" (e.g., "0.6:0.4") |
| --scope | -sc | No | global | Percentile scope (global, sub_family) |
| --skill-column | -sk | No | skill_name | Column name for skill identifier |
| --thresholds | -t | No | N/A | Custom thresholds JSON file path |
| --include-partial | -ip | No | False | Include partial matches with flags |
| --reconciliation-report | -rr | No | False | Export reconciliation report |
| --sensitivity | N/A | No | False | Run sensitivity analysis |
| --summary | N/A | No | False | Generate summary statistics |
| --verbose | -v | No | False | Enable verbose logging |

### Example Commands

**Basic run:**

```bash
python sunrise_sunset_classifier.py \
  -s supply_data.csv \
  -dc demand_2025.csv \
  -dp demand_2024.csv \
  -o results.csv
```

**With custom weights and reconciliation report:**

```bash
python sunrise_sunset_classifier.py \
  -s supply_data.csv \
  -dc demand_2025.csv \
  -dp demand_2024.csv \
  -o results.csv \
  -cw "0.6:0.4" \
  --reconciliation-report
```

**With sub-family scope and sensitivity analysis:**

```bash
python sunrise_sunset_classifier.py \
  -s supply_data.csv \
  -dc demand_2025.csv \
  -dp demand_2024.csv \
  -o results.json \
  -f json \
  -sc sub_family \
  --sensitivity
```

**Full featured run:**

```bash
python sunrise_sunset_classifier.py \
  --supply supply_data.csv \
  --demand-current demand_2025.csv \
  --demand-previous demand_2024.csv \
  --output results.xlsx \
  --format xlsx \
  --weights supply_constrained \
  --scope sub_family \
  --include-partial \
  --reconciliation-report \
  --sensitivity \
  --summary \
  --verbose
```

---

## 16. API Specification (Optional)

If implementing as a service:

### Endpoint

```
POST /api/v1/classify
```

### Request Body

```json
{
  "supply": [
    {
      "skill_name": "Microarchitecture",
      "sub_family": "Silicon & SoC Design",
      "talent_pool_size": 2400
    }
  ],
  "demand_current": [
    {
      "skill_name": "Microarchitecture",
      "job_openings_current": 4600
    }
  ],
  "demand_previous": [
    {
      "skill_name": "Microarchitecture",
      "job_openings_previous": 3800
    }
  ],
  "config": {
    "weights": "balanced",
    "scope": "global",
    "include_partial": false,
    "include_reconciliation": true,
    "include_summary": true
  }
}
```

### Response Body

```json
{
  "status": "success",
  "reconciliation": {
    "supply_total": 500,
    "demand_current_total": 500,
    "demand_previous_total": 500,
    "common_skills_count": 412,
    "match_rate_pct": 75.2
  },
  "results": [
    {
      "skill_name": "Microarchitecture",
      "sunrise_score": 100.0,
      "classification": "Sunrise",
      "confidence": "High"
    }
  ],
  "summary": {
    "total_skills_analyzed": 412,
    "classification_distribution": {
      "Sunrise": 58,
      "Likely Sunrise": 92,
      "Neutral/Transitional": 84,
      "Likely Sunset": 101,
      "Sunset": 77
    }
  },
  "metadata": {
    "processed_at": "2025-02-18T10:30:00Z",
    "version": "1.0.0"
  }
}
```

---

## 17. Testing Requirements

### Unit Tests

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| Basic classification | Valid skill data | Correct classification |
| Edge case: zero previous demand | demand_previous_year = 0 | confidence = Low |
| Edge case: zero current demand | demand_current_year = 0 | classification = Sunset |
| Edge case: single skill | One skill in dataset | score = 50, classification = Neutral |
| Custom weights | weights = 0.6:0.4 | Correct weighted score |
| Tied ranks | Two skills with same growth | Average rank assigned |
| Skill normalization | "ASIC Design" vs "asic design" | Match successfully |
| No common skills | Disjoint skill sets | Raise error E007 |
| Partial match handling | Skill in 2 of 3 tables | Flag appropriately |

### Integration Tests

| Test Case | Description |
|-----------|-------------|
| Three file input processing | Read and merge three CSV files correctly |
| Reconciliation report generation | Generate accurate match statistics |
| JSON input processing | Read and process JSON files correctly |
| Multi-format output | Generate CSV, JSON, XLSX outputs correctly |
| Large dataset handling | Process 500+ skills per table efficiently |
| Sub-family scoping | Calculate percentiles within sub-families |
| Partial match inclusion | Include and flag partial matches correctly |

---

## 18. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-02-18 | Initial release |
| 1.1.0 | 2025-02-18 | Added data reconciliation step for multiple input tables |

---

## 19. Appendix: Worked Example

### Input Data (Three Tables)

**Supply Table:**

| Skill | Sub-Family | Talent Pool |
|-------|------------|-------------|
| Microarchitecture | Silicon & SoC Design | 2,400 |
| ASIC Design | Silicon & SoC Design | 1,800 |
| Firmware Development | Embedded Software | 12,000 |
| Test Planning | Test & Compliance | 8,500 |
| RTOS Development | Embedded Software | 3,200 |
| PCB Design | Electronics | 6,500 |
| DSP | RF & Wireless | 2,100 |
| Device Drivers | Embedded Software | 4,800 |

**Demand Current Year Table:**

| Skill | Job Openings |
|-------|--------------|
| Microarchitecture | 4,600 |
| ASIC Design | 2,350 |
| Firmware Development | 10,200 |
| Test Planning | 4,100 |
| RTOS Development | 3,300 |
| PCB Design | 3,650 |
| DSP | 2,900 |
| Device Drivers | 3,750 |

**Demand Previous Year Table:**

| Skill | Job Openings |
|-------|--------------|
| Microarchitecture | 3,800 |
| ASIC Design | 2,200 |
| Firmware Development | 9,500 |
| Test Planning | 4,200 |
| RTOS Development | 2,800 |
| PCB Design | 3,800 |
| DSP | 2,400 |
| Device Drivers | 3,600 |

### Step 0: Reconciliation

All 8 skills match across all 3 tables. Match rate = 100%.

### Step 1: Calculate Core Metrics

| Skill | Demand Growth Rate | Supply Pressure Index |
|-------|--------------------|-----------------------|
| Microarchitecture | +21.1% | 0.52 |
| ASIC Design | +6.8% | 0.77 |
| Firmware Development | +7.4% | 1.18 |
| Test Planning | -2.4% | 2.07 |
| RTOS Development | +17.9% | 0.97 |
| PCB Design | -3.9% | 1.78 |
| DSP | +20.8% | 0.72 |
| Device Drivers | +4.2% | 1.28 |

### Step 2: Convert to Percentiles

| Skill | Demand Growth | Rank | Demand Pctl | Supply Pressure | Rank | Supply Pctl |
|-------|---------------|------|-------------|-----------------|------|-------------|
| Microarchitecture | +21.1% | 1 | 100.0 | 0.52 | 1 | 100.0 |
| DSP | +20.8% | 2 | 87.5 | 0.72 | 2 | 87.5 |
| RTOS Development | +17.9% | 3 | 75.0 | 0.97 | 4 | 62.5 |
| Firmware Development | +7.4% | 4 | 62.5 | 1.18 | 5 | 50.0 |
| ASIC Design | +6.8% | 5 | 50.0 | 0.77 | 3 | 75.0 |
| Device Drivers | +4.2% | 6 | 37.5 | 1.28 | 6 | 37.5 |
| Test Planning | -2.4% | 7 | 25.0 | 2.07 | 8 | 12.5 |
| PCB Design | -3.9% | 8 | 12.5 | 1.78 | 7 | 25.0 |

### Step 3: Calculate Sunrise Score (Balanced Weights)

| Skill | Demand Pctl | Supply Pctl | Sunrise Score |
|-------|-------------|-------------|---------------|
| Microarchitecture | 100.0 | 100.0 | **100.0** |
| DSP | 87.5 | 87.5 | **87.5** |
| RTOS Development | 75.0 | 62.5 | **68.8** |
| ASIC Design | 50.0 | 75.0 | **62.5** |
| Firmware Development | 62.5 | 50.0 | **56.3** |
| Device Drivers | 37.5 | 37.5 | **37.5** |
| PCB Design | 12.5 | 25.0 | **18.8** |
| Test Planning | 25.0 | 12.5 | **18.8** |

### Step 4: Final Classification

| Skill | Sunrise Score | Classification | Confidence |
|-------|---------------|----------------|------------|
| Microarchitecture | 100.0 | Sunrise | High |
| DSP | 87.5 | Sunrise | High |
| RTOS Development | 68.8 | Likely Sunrise | High |
| ASIC Design | 62.5 | Likely Sunrise | High |
| Firmware Development | 56.3 | Neutral/Transitional | High |
| Device Drivers | 37.5 | Likely Sunset | High |
| PCB Design | 18.8 | Sunset | High |
| Test Planning | 18.8 | Sunset | High |

---

**End of Documentation**
