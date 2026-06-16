# GenHack 4 - 2025 - Project: Urban Heat Island Bias Correction for ERA5-Land Analysis

[Final Report](https://drive.google.com/file/d/11FVX2S-9z6TcmY3-bj92UHgDHBYjnotg/view?usp=sharing)

---

### The Challenge
Urban Heat Islands (UHI) represent one of the most significant anthropogenic modifications to local climate, with profound implications for public health, energy consumption, and urban resilience. The **GenHack 4** challenge focused on analyzing and correcting systematic biases in ERA5-Land reanalysis data when applied to urban environments, using Madrid, Spain as a primary case study.

### Core Problem
ERA5 reanalysis data, while comprehensive and globally consistent, systematically overestimates temperatures in urban areas due to its inability to resolve fine-scale urban processes. This bias amplifies during heatwaves, creating significant errors for climate adaptation planning and heat-health warning systems.

---

## Our Unique Approach

### Three-Pillar Methodology

#### 1. **Comprehensive Urban Climate Analysis**
- **Multi-source data integration**: ERA5-Land, ECA&D station data, Sentinel-2 NDVI, WUDAPT LCZ classifications
- **Dual UHI quantification methods**: LCZ-based (primary) and distance-based (fallback) approaches
- **Temporal analysis**: 2020-2025 period with daily resolution
- **Validation suite**: 12+ statistical metrics including temperature-regime specific biases

#### 2. **Machine Learning Ensemble Framework**
We developed a **four-model ensemble** specifically designed to address different aspects of urban climate modeling:

| Model | Purpose | Key Innovation |
|-------|---------|----------------|
| **XGBoost Bias Correction** | Corrects systematic ERA5 biases | Nonlinear, feature-dependent correction |
| **Random Forest UHI Prediction** | Predicts UHI intensity directly | Exceptional generalization (0.009 R² gap) |
| **Ridge Regression Downscaling** | Statistical downscaling of ERA5 | Interpretable coefficients for physical understanding |
| **Quantile Regression** | Uncertainty quantification | Full prediction intervals for risk assessment |

#### 3. **Interactive Visualization Platform**
- **Full-stack Flask API** with comprehensive endpoints
- **Real-time multi-city analysis** with KNN clustering
- **Dynamic visualization system** supporting 10+ chart types
- **Downloadable data and metrics** for reproducibility

---

## Key Scientific Findings

### Madrid's Urban Climate Signature (2020-2025)

#### Temperature Validation Results
- **Maximum Temperature (TX)**: Excellent agreement with ERA5 (R²=0.992, bias=+0.34°C)
- **Minimum Temperature (TN)**: Significant warm bias (+1.95°C), amplified during heatwaves (+2.99°C)
- **Nocturnal UHI Intensity**: Mean = 2.25°C, extreme events up to 7.76°C

#### Climate Extremes Profile
- **Heatwaves**: 30 events/year, average duration 7.8 days, maximum 43 days
- **Cold spells**: 19 events/year, urban moderation effect evident
- **Precipitation extremes**: 99th percentile threshold = 17.82 mm/day

#### Environmental Relationships
- **NDVI cooling elasticity**: -16.52°C per NDVI unit (strong theoretical cooling potential)
- **Wind-temperature correlation**: Weak (-0.12), suggesting radiative processes dominate
- **Pressure-temperature relationship**: Strong (-0.33 to -0.39), indicating synoptic control

---

## Technical Innovations

### Methodological Advances

#### 1. **Anti-Overfitting-First ML Design**
- Exhaustive grid search (48-50 hyperparameter combinations per model)
- Conservative parameter selection prioritizing generalization
- Cross-validation with multiple metrics ensuring robustness

#### 2. **Multi-City Comparative Analysis Framework**
- KNN clustering for climate similarity assessment
- Standardized feature extraction across cities
- Geographic mapping with interactive bias visualization

#### 3. **Urban-Specific Feature Engineering**
- Harmonic encoding for seasonal cycles
- LCZ-informed urban morphology features
- Residual learning for bias correction

### Computational Architecture

```
Frontend (React/Vue.js) 
    ↓
API Layer (Flask + 25+ endpoints)
    ↓
Data Processing (Pandas/NumPy/SciPy)
    ↓
Machine Learning (Scikit-learn/XGBoost)
    ↓
Data Storage (CSV/JSON + metadata)
```

---

## Impact and Applications

### For Climate Science
1. **Validated ERA5 limitations** specifically for urban Mediterranean climates
2. **Quantified temperature-regime dependent biases** essential for bias correction
3. **Demonstrated ML superiority** over traditional statistical methods for urban climate modeling

### For Urban Planning
1. **Actionable UHI metrics** for heat mitigation strategies
2. **Vegetation cooling potential quantification** supporting green infrastructure planning
3. **Extreme event characterization** for climate adaptation planning

### For Public Health
1. **Heatwave prediction improvement** through bias-corrected temperature forecasts
2. **Nighttime temperature accuracy** critical for heat-health warning systems
3. **Uncertainty quantification** enabling risk-based decision making

---


## Results and Metrics

### Model Performance Summary

| Model | CV RMSE | CV R² | Generalization Gap |
|-------|---------|-------|-------------------|
| **Random Forest UHI** | 0.21°C | 0.737 | 0.009 (excellent) |
| **Ridge Regression** | 1.88°C | 0.893 | 0.099 (good) |
| **XGBoost Bias Correction** | 2.78°C | 0.886 | 0.240 (moderate) |
| **Quantile Regression** | - | - | 68.3% coverage (80% target) |

### Key Performance Indicators
- **UHI prediction accuracy**: ±0.21°C (best in class for daily station-based prediction)
- **Bias reduction**: 11% improvement over raw ERA5 for minimum temperatures
- **Computational efficiency**: All models train in <30 minutes, predict in <1 second
- **Scalability**: Framework extensible to 100+ European cities

---

## Future Directions

### Immediate Extensions
1. **Multi-city deployment**: Apply framework to 50+ European cities
2. **Real-time API**: Operational deployment for municipal heat forecasting
3. **Climate projections**: Apply bias correction to CMIP6 future scenarios

### Research Advancements
1. **Causal discovery**: Move beyond correlation to identify causal drivers
2. **Spatiotemporal modeling**: Graph neural networks for station networks
3. **Transfer learning**: Knowledge sharing between climate-similar cities

### Policy Integration
1. **Heat action plans**: Integrate with municipal heat-health warning systems
2. **Urban design guidelines**: Evidence-based recommendations for cooling strategies
3. **Climate resilience indicators**: Standardized metrics for urban climate assessment

---

## Why Our Project Stands Out

### Scientific Rigor
- **Multi-method validation** against ground truth station data
- **Statistical robustness** with comprehensive error metrics
- **Physical interpretability** maintained despite ML complexity

### Technical Excellence
- **Production-ready codebase** with full documentation
- **Scalable architecture** supporting multi-city analysis
- **Interactive visualization** enabling exploratory analysis

### Real-World Impact
- **Actionable insights** for urban heat mitigation
- **Policy-relevant metrics** for climate adaptation
- **Health-protective applications** through improved temperature forecasts

---

## References and Acknowledgements

### Data Sources
- **ERA5-Land**: Copernicus Climate Change Service
- **ECA&D Station Data**: European Climate Assessment & Dataset
- **Sentinel-2 NDVI**: European Space Agency
- **WUDAPT LCZ**: World Urban Database and Access Portal Tools

### Technical Stack
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Plotly, Matplotlib
- **Backend**: Flask, REST API
- **Frontend**: React, D3.js

### Acknowledgments
We extend our gratitude to the **GenHack 2025 organizing committee**, **BNP Paribas "Stress Test" Chair**, and **École Polytechnique** for providing this exceptional platform for climate innovation.

---

*"Advancing urban climate science through innovative data integration and machine learning, creating actionable insights for a more resilient urban future."*

---
**Hackathon**: GenHack 4 2025
