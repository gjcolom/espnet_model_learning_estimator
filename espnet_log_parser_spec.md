# ESPnet Training Log Parser & Grapher — High-Level Specification (Editor Mode Document)
*Conceptual behavior only — no implementation details.*

---

# **1. Purpose**

The tool must read ESPnet training logs and generate visual graphs that assist in evaluating:

- Model convergence  
- Training vs. validation performance  
- Diminishing returns  
- Overfitting risks  
- Optimal stopping epoch  

It must accept logs from any ESPnet version and produce visually interpretable analytics.

---

# **2. Input Requirements**

## **2.1 Accepted Inputs**
The tool must accept:

- A path to **one or more** ESPnet training log files  

These logs may include:

- Per-epoch training loss  
- Per-epoch validation loss  
- Additional auxiliary losses (optional)

## **2.2 Flexible Log Recognition**
The parser must:

- Detect epoch numbers  
- Detect training loss  
- Detect validation loss  
- Ignore irrelevant lines  
- Handle logs with varying formatting or verbosity  

No assumptions are made about the exact structure beyond standard ESPnet output patterns.

---

# **3. Data Extraction Requirements**

## **3.1 Extracted Metrics**
For each epoch, the tool must extract:

- `epoch`  
- `train_loss`  
- `valid_loss`  
- Optional: any consistent auxiliary metrics (e.g., duration, pitch, energy losses)

## **3.2 Missing Data Handling**
If expected values are missing:

- Skip incomplete epochs  
- Do not interrupt processing  
- Optionally note warnings in output  

---

# **4. Graphing Requirements**

The tool must generate **three primary graph types**.

---

## **4.1 Linear-Scale Loss Graph**
A graph showing:

- Training loss vs epoch  
- Validation loss vs epoch  
- Linear Y-axis  

Purpose:  
To visualize broad convergence trends across training.

---

## **4.2 Logarithmic-Scale Loss Graph**
Contains the same data as the linear graph, but:

- Y-axis is logarithmic  

Purpose:  
To reveal small improvements that flatten on linear scale.

---

## **4.3 Improvement Graph (Epoch vs Validation Loss Delta)**  
*(New requirement)*

The tool must produce a graph where:

### **X-axis:**  
- Epoch number `n`

### **Y-axis:**  
- Improvement value defined as:  
```
improvement[n] = valid_loss[n-1] - valid_loss[n]
```

### **Purpose:**  
- To visualize diminishing returns  
- To identify plateau regions  
- To support early-stopping heuristics  
- To show the exact point where model improvement becomes negligible  

### **Behavior:**  
- Positive values = improvement  
- Near-zero values = plateau  
- Negative values = regression  

### **Optional Features (nice-to-have):**  
- Logarithmic scale version of the improvement curve  
- Smoothing window (e.g., rolling average)  
- A threshold reference line (e.g., 0.001)  
- Highlighting epochs where improvement falls below threshold  
- Automatically marking the “first plateau epoch”  

---

# **5. Output Requirements**

The tool must produce:

## **5.1 Graph Files**
- Linear loss curve image  
- Log-scale loss curve image  
- Improvement curve image  

Formats may include PNG, PDF, or SVG.

## **5.2 Optional Combined Output**
A single figure containing:

- Linear loss curve  
- Log loss curve  
- Improvement curve  

---

# **6. Diminishing Returns Analysis**

The tool must compute simple indicators that help identify plateau behavior.

## **6.1 Epoch-to-Epoch Loss Delta**
Compute:

```
delta[n] = valid_loss[n-1] - valid_loss[n]
```

## **6.2 Threshold-Based Plateau Detection**
The tool must allow the user to specify:

- A delta threshold (e.g., 0.001 absolute or relative)  
- A consecutive-epoch count to confirm diminishing returns  

If improvement remains below the threshold for N consecutive epochs, the tool must identify this epoch range as a plateau.

## **6.3 Suggested Stop Epoch (Heuristic)**
The tool must output a recommended stopping point based on:

- Earliest plateau epoch  
- Validation loss stability  
- Minimal improvements beyond a set threshold  

This is a **statistical heuristic**, not a machine-learning algorithm.

---

# **7. User Interaction Requirements**

## **7.1 Tool Must Be Interface-Agnostic**
The specification must support:

- CLI tools  
- Future GUI integration  
- Use in standalone scripts  

## **7.2 Configurable Options**
User must be able to specify:

- Log file path  
- Output directory  
- Delta threshold  
- Number of epochs needed to confirm diminishing returns  
- Whether graphs should display interactively or only export  

---

# **8. General Constraints**

## **8.1 No Version Assumptions**
Tool must work with ESPnet:

- v1  
- v2  
- Any comparable log format  

## **8.2 Robust Parsing**
Parser must tolerate:

- Irregular spacing  
- Extra log noise  
- Additional metrics  
- Progress meter lines  
- Partial epochs  

## **8.3 Cross-Platform Compatible**
Design must not rely on:

- OS-specific behavior  
- Path conventions  
- Implementation-specific libraries  

---

# **9. Deliverables (Conceptual)**

The final tool must output:

- Parsed epoch-wise data structure  
- Linear-scale training/validation loss graph  
- Log-scale training/validation loss graph  
- Epoch vs loss-delta improvement graph  
- Summary statistics of training behavior  
- Plateau detection results  
- Suggested early-stop epoch (heuristic)  

All without prescribing any implementation language, libraries, or UI framework.

---

# 📌 **End of Editor Mode Specification**  
