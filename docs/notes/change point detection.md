---
tags:
- Statistics
- time series
title: change point detection
categories:
date: 2022-12-26
lastMod: 2022-12-26
---
# Overview


  + **Change point detection**, also known as **change point analysis**, is a statistical technique used to identify points in a time series where the statistical properties of the data change. These points, known as *change points*, can correspond to shifts, trends, or other types of changes in the data.

  + Change point detection is often used in a variety of applications, including financial analysis, manufacturing, and quality control, to identify changes in the underlying processes generating the data and to improve forecasting accuracy.

  + There are several different methods for detecting change-points, including both *parametric* and *non-parametric* approaches.

    + *Parametric methods* assume that the data follows a specific statistical distribution, such as a normal distribution, and use this assumption to identify change-points.

    + *Non-parametric methods*, on the other hand, do not make any assumptions about the underlying distribution of the data and are generally more flexible and robust.

  + To detect change-points, analysts typically start by dividing the data into segments and then testing each segment to see if it differs significantly from the surrounding segments. If a segment is found to be significantly different, it is marked as a change-point. This process is typically repeated until all changepoints have been identified.

  + Once change-points have been identified, analysts can use them to improve forecasting accuracy by fitting separate models to each segment of the data and then using these models to make predictions about future values of the time series. They can also use the change-points to understand the underlying processes generating the data and to identify possible causes for the changes that occurred at the change-points.

  + Change point methods can be either *offline or online*; *parametric or nonparametric*.

  + ## Parametric Change-point Models

    + **CUSUM:** use a cumulative sum of the deviations from the mean to identify change-points.

    + **EWMA:** use a weighted average of past values to identify change-points. [^1]

    + **ARIMA:** use a combination of autoregressive and moving average terms to identify change-points.

  + ## Non-Parametric Change-point Models

    + **Binary segmentation:** This method divides the data into segments and tests each segment to see if it differs significantly from the surrounding segments.

    + **PELT (Pruned Exact Linear Time):** uses a dynamic programming approach to identify change-points in a time series.

    + **Wild Binary Segmentation:** This method is similar to binary segmentation, but allows for multiple change-points within each segment.

    + Change-point methods based on **penalized likelihoods**: These methods use a penalized likelihood approach to identify change-points in the data.

  + ## Other Approaches

    + There are also *hybrid* change-point detection models that combine elements of parametric and non-parametric approaches.

# Core Concepts

  + **Minimum Description Length (MDL)** pertains to the speed of detection of a change-point

  + 



# References

  + [^1]:  [Univariate to Multivariate EWMA Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc343.htm)

  + 
