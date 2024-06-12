# AI: Predicting Global Temperature Change Using Neural Networks

### Code Description
[Link to Tutorial](Tutorial.md)


## Introduction

This project focuses on predicting global temperature changes using a simple neural network with historical data. Traditional approaches to climate prediction typically require sophisticated modeling and significant computing power. However, neural networks can learn to model the global climate by training on properly formatted historical data. Moreover, for long-term predictions, sophisticated approaches may not be very efficient. Therefore, we employ simple neural network models to provide cost-effective and time-efficient climate modeling for long-term predictions.

## Data Sources

1. **Global Monthly 1° x 1° Latitude-Longitude Grid**
   - **Source**: Berkeley Earth
   - **Details**: Temperature; Land + Ocean Average Temperature with Air Temperatures at Sea Ice
   - **Time Period**: Monthly data from 1850-2023
   - **Access**: [Berkeley Earth Temperature Data](https://berkeleyearth.org/data/)

2. **Nature Scientific Data (CO2 PPM)**
   - **Source**: Nature Scientific Data
   - **Details**: Global monthly gridded atmospheric carbon dioxide concentrations under the historical and future scenarios
   - **Time Period**: Monthly historical data from 1850-2013, and Monthly SSP scenarios from 2015-2150
   - **Access**: [Nature Scientific Data on CO2 PPM](https://www.nature.com/articles/s41597-022-01196-7)


## Data
### Variables

| Variable | Description |
| --- | --- |
| Temperature | Deviation from 1951-1980 monthly average, in °C |
| CO2 | Concentration in parts per million (PPM) |
| Land Mask (land ocean ratio) | Range (0, 1) |
| Latitude (-90° to 90°) (Normalized) | Range (-1, 1) |
| Month (1-12) (Cosine Normalized) | Range (-1, 1) |
| Month (1-12) (Sine Normalized) | Range (-1, 1) |

- Relying solely on temperature data lacks the predictive power for future conditions.
- To enhance prediction capabilities, greenhouse gas (GHG) data have been incorporated.

### Dataset
![Picture1](https://github.com/ClimatePrediction2100/ai/assets/70141850/d28ca348-a0cf-49af-9018-55487b18d7e4)

**Train Set**: 1850-2013 Random* Sampling (Max 400k samples, Early Stop applied)

**Test Set**: 2015-2023 Random* Sampling (1k samples)

**Prediction Set**: 2024-2100 AI Simulation (Sliding Window applied)

- Samples are randomly picked by Time, Latitude, Longitude, and are resampled if NaN values are included.

### Data Sample
![Picture2](https://github.com/ClimatePrediction2100/ai/assets/70141850/d2230516-4f89-4933-9014-d996626c9279)

- By training with randomly extracted 1° x 1° area data, we designed a generalized model that mitigates temporal and regional biases.
- The sample's Time Step length is set to a maximum of 48 months, designing a predictive model based on the current state without reflecting long-term trends.

## Models
### Model Comparison

Comparison experiments of five model types:
- MLP (Multi-Layer Perceptron)
- RNN (Recurrent Neural Network)
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)
- LSTM + Attention

Each model undergoes training and evaluation across 108 different weight configurations.

### Hyperparameters

| Parameter       | Options                                |
|-----------------|----------------------------------------|
| Layers          | 2, 4, 6                                |
| Hidden Dimension| 100, 200                               |
| Cost Function   | MSE (Mean Squared Error), MAE (Mean Absolute Error), Huber |
| Learning Rate   | 0.01, 0.001                            |
| Sequence Length | 12, 24, 48 (Time Steps = Months)       |

**Model and Hyperparameter Selection Considerations**

After comprehensive consideration of error and model size, the selected configuration includes:
- **Model Type**: LSTM
- **Layers**: 2
- **Hidden Dimension**: 100
- **Loss Function**: Huber
- **Learning Rate**: 0.001
- **Sequence Length**: 48 months

## Simulation

<div align="center">
  <a href="https://youtu.be/IabjwhqKuio" target="_blank">
    <img src="http://img.youtube.com/vi/IabjwhqKuio/0.jpg" alt="Data Visualization">
  </a>
  <p><a href="https://youtu.be/IabjwhqKuio" target="_blank">Result Visualziation Video</a></p>
</div>



**Comparison with IPCC 6th Assessment Report AR6 WG1 & WG2 and Simulation Results**

**(Relative to the global average temperature from 1850-1900, based on SSP scenarios, in °C)**

| Scenario | 2041-2060 (From report) | 2050 (Simulation) | 2081-2100 (From report) | 2090 (Simulation) |
| --- | --- | --- | --- | --- |
| SSP1-1.9 | 1.2 - 2.0 | 1.97 | 1.0 – 1.8 | 1.29 |
| SSP1-2.6 | 1.3 - 2.2 | 2.34 | 1.3 – 2.4 | 2.26 |
| SSP2-4.5 | 1.6 - 2.5 | 2.55 | 2.1 – 3.5 | 2.79 |
| SSP3-7.0 | 1.7 - 2.6 | 2.66 | 2.8 – 4.6 | 2.97 |
| SSP5-8.5 | 1.9 - 3.0 | 2.71 | 3.3 – 5.7 | 3.05 |

**Annual Average Temperature Changes According to SSP (Shared Socioeconomic Pathways)**

**(Relative to the global average temperature from 1850-1900, for the period 2024-2100, across five scenarios, in °C)**
![Picture3](https://github.com/ClimatePrediction2100/ai/assets/70141850/75adff45-2ad7-4aad-a3c6-a7fa9aeeda78)


## Conclusion

- This project introduces a novel approach beyond traditional modeling techniques.
- The model training utilized up to 400,000 data samples, verifying that effective modeling can be achieved with a relatively small amount of data.
- Similar results to those in the IPCC's Sixth Assessment Report validate the effectiveness of the proposed methods.

## References

1. Cheng, W., et al. (2022). Global monthly gridded atmospheric carbon dioxide concentrations under the historical and future scenarios. *Scientific Data*, *9*(1). [https://doi.org/10.1038/s41597-022-01196-7](https://doi.org/10.1038/s41597-022-01196-7)

2. IPCC, 2021: Summary for Policymakers. In: *Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change*. [https://dx.doi.org/10.1017/9781009157896.001](https://dx.doi.org/10.1017/9781009157896.001)

3. Riahi, K., et al. (2017). The shared socioeconomic pathways and their energy, land use, and greenhouse gas emissions implications: An overview. *Global Environmental Change*, *42*, 153–168. [https://doi.org/10.1016/j.gloenvcha.2016.05.009](https://doi.org/10.1017/9781009157896.001)

4. Rohde, R. A., & Hausfather, Z. (2020). The Berkeley Earth Land/Ocean Temperature Record. *Earth System Science Data*, *12*(4), 3469–3479. [https://doi.org/10.5194/essd-12-3469-2020](https://doi.org/10.5194/essd-12-3469-2020)


