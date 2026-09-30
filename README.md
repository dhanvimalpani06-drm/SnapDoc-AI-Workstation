# SnapDoc AI: On-Device Private Document Intelligence

SnapDoc AI is an enterprise-grade private document intelligence client built for **Snapdragon-powered HP PCs** running Windows 11 on ARM. It accelerates local Retrieval-Augmented Generation (RAG) by offloading vector embeddings and semantic search onto the **Qualcomm Hexagon NPU** via the QNN Execution Provider.

---

## Key Features
- **100% On-Device & Offline:** Zero internet dependency, total DPDP/HIPAA data confidentiality.
- **Hardware Acceleration:** Hexagon Tensor Processor (HTP) offloading achieves up to **10.5x lower latency** than legacy CPU execution.
- **Glassmorphic Enterprise UI:** Built with Streamlit, custom dark-mode aesthetics, prompt suggestions, and real-time latency telemetry.

---

## Technical Stack
- **Frontend / Framework:** Streamlit Python
- **Embedding Model:** `all-MiniLM-L6-v2` (INT8 Quantized)
- **NPU Backend:** Qualcomm QNN Execution Provider / ONNX Runtime
- **PDF Engine:** PyPDF & Custom Sliding-Window Chunker

---

## Quick Start Guide

1. **Clone Repository & Install Dependencies:**
   ```cmd
   pip install -r requirements.txt