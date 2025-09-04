# March Madness Tournament Simulator

## Table of Contents
1. [Abstract](#abstract)
2. [Features](#features)
3. [Results](#results)
4. [Future Work](#future-work)
5. [Acknowledgments](#acknowledgments)
6. [Website](#website)

---

## Abstract
This is a **March Madness Tournament Simulator** designed to predict NCAA basketball tournament outcomes using historical performance data and machine learning.  

The simulator is built to run annually, updating predictions, region strength tables, and visualizations each year. While the current version highlights the **2024 tournament**, the framework supports **any tournament year** by rerunning the data pipeline and prediction scripts.  

The primary objectives of the project are:  
- To predict game outcomes using statistical and machine learning models.  
- To simulate full tournament brackets and calculate team advancement probabilities.  
- To evaluate and visualize region strength, potential upsets, and Final Four odds.  

---

## Features
- **Data Pipeline**: Aggregates NCAA performance data from multiple sources.  
- **Predictive Modeling**: Uses random forest models to simulate game outcomes.  
- **Tournament Simulation**: Runs full tournament simulations to estimate team advancement odds.  
- **Region Strength Analysis**: Calculates and visualizes region strength to identify seeding imbalances.  
- **Visualizations**: Generates interactive and static outputs, including heatmaps and probability charts.  
- **Annual Updates**: Designed to refresh predictions for each new NCAA tournament year.  

---

## Results
- **2024 Predictions**: Produced full tournament simulations, Final Four odds, and mis-seeding analysis.  
- **Region Strength Tables**: Identified relative imbalances across tournament regions.  
- **Visualization Outputs**: Generated heatmaps and charts to display probabilities of team advancement.  

---

## Future Work
- Automate yearly updates for each NCAA tournament.  
- Explore additional modeling approaches (e.g., gradient boosting, neural networks).  
- Integrate betting line comparisons for calibration and validation.  
- Add interactive web visualizations for live bracket exploration.  

---

## Acknowledgments
- NCAA and sports reference datasets.  
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`.  

---

## Website
This project is also featured on my personal website, with region strength tables and tournament visualizations:  
[Visit the Project Page](https://txcwalker.github.io/projects/marchmadness/)

