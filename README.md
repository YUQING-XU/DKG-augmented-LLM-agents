# DKG-augmented-LLM-agents

This repository contains the source code for the paper *"Exploring Automated Carbon-aware Assessment for openBIM-based Ductwork Design Using KG-augmented LLM Multi-agents"*, implementing a Knowledge Graph (KG)-augmented LLM multi-agent framework for embodied carbon assessment and optimization of ductwork systems.

## 🚀 Key Features

- **Input Processing**:
  - IFC file of the illustrative example's ductwork system
  - Catalogs of the ductwork components
- **Knowledge Graph Integration**:
  - IFC extraction and KG construction scripts (IFC -> DKG)
  - JSON file of the genertaed DKG
  - PNG of the DKG visualization
- **Three Specialized LLM Agents**:
  - `CalculationAgent`: Accurate embodied carbon calculation
  - `RecommendationAgent`: Material selection for carbon-cost trade-offs
  - `CheckingAgent`: Air distrbution compliance checking

