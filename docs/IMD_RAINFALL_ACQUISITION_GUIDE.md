# IMD Rainfall Data Acquisition & Feature Engineering Guide

**Project:** NER-Nav  
**Purpose:** Guide acquisition, ingestion, and processing of real rainfall data for road-disruption hazard modeling  
**Status:** Ingestion Plan - P0 Priority

---

## 1. Requirements

Rainfall is the primary triggering factor for landslides, floods, and road disruptions in NER.

**Target Period:** 2015-2024  
**Target Region:** 8 NER states, 131 districts  
**Resolution:** Daily minimum (hourly preferred)

---

## 2. Data Source

**Provider:** India Meteorological Department (IMD)  
**URL:** https://mausam.imd.gov.in/  
**License:** OGD India  
**Dataset:** Daily Gridded Rainfall (0.25° resolution)

---

## 3. Next Actions

1. Register at IMD portal for dataset access
2. Download NetCDF files for NER region (2015-2024)
3. Store in `data/raw/weather/`
4. Implement temporal-leakage-safe feature engineering
